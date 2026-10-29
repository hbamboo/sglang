"""Customer multi-request prefill bursts, each capped at 32768 new tokens."""
import argparse
import gzip
import hashlib
import json
import pathlib
import statistics
import time

import requests
from transformers import AutoTokenizer

parser=argparse.ArgumentParser()
parser.add_argument('--base',default='http://127.0.0.1:19255')
parser.add_argument('--dataset',default='/mnt/sfs_turbo/hxd/CHAT-mxfp8-customer-2000.jsonl')
parser.add_argument('--output',required=True)
parser.add_argument('--cache',default='/mnt/paas/hxdnpu/glm-shared-overlap-20261030/customer-inputs.json.gz')
parser.add_argument('--profile-dir')
parser.add_argument('--profile-pack',type=int,default=0)
args=parser.parse_args()
dataset=pathlib.Path(args.dataset)
identity=hashlib.sha256(dataset.read_bytes()).hexdigest()
cache=pathlib.Path(args.cache)
if cache.exists():
    with gzip.open(cache,'rt') as file:data=json.load(file)
    assert data['dataset_sha256']==identity
else:
    samples=[json.loads(line) for line in dataset.open() if line.strip()]
    tokenizer=AutoTokenizer.from_pretrained('/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8',trust_remote_code=True)
    texts=[sample['conversations'][0]['content'] for sample in samples]
    ids=tokenizer(texts,add_special_tokens=True)['input_ids']
    data=dict(dataset_sha256=identity,input_ids=ids)
    with gzip.open(cache,'wt') as file:json.dump(data,file)
assert len(data['input_ids'])==2000
packs=[]
current=[]
tokens=0
for index,ids in enumerate(data['input_ids']):
    assert 0<len(ids)<=32768, (index,len(ids))
    if current and (tokens+len(ids)>32768 or len(current)==32):
        packs.append(current);current=[];tokens=0
    current.append(index);tokens+=len(ids)
if current:packs.append(current)
print('DATASET',json.dumps(dict(requests=2000,tokens=sum(map(len,data['input_ids'])),packs=len(packs),min_pack_tokens=min(sum(len(data['input_ids'][i]) for i in p) for p in packs),max_pack_tokens=max(sum(len(data['input_ids'][i]) for i in p) for p in packs),max_pack_requests=max(map(len,packs)),dataset_sha256=identity)),flush=True)
session=requests.Session()

def run_pack(pack):
    prompts=[data['input_ids'][i] for i in pack]
    assert sum(map(len,prompts))<=32768
    payload=dict(input_ids=prompts,stream=True,sampling_params=dict(temperature=0,max_new_tokens=1,ignore_eos=True))
    start=time.perf_counter();first={};responses={}
    with session.post(args.base+'/generate',json=payload,stream=True,timeout=900) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b'data: '):continue
            raw=line[6:]
            if raw==b'[DONE]':break
            item=json.loads(raw)
            if item.get('meta_info',{}).get('completion_tokens',0)<1:continue
            index=item.get('index',0)
            if index not in first:first[index]=(time.perf_counter()-start)*1000
            responses[index]=item
    assert len(responses)==len(pack),(len(responses),len(pack))
    records=[]
    for index,original in enumerate(pack):
        item=responses[index];meta=item['meta_info']
        assert meta['prompt_tokens']==len(prompts[index]) and meta['completion_tokens']==1 and meta.get('cached_tokens',0)==0,meta
        records.append(dict(dataset_index=original,input_tokens=len(prompts[index]),ttft_ms=first[index],output_ids=item['output_ids'],meta_info=meta))
    return dict(requests=len(pack),tokens=sum(map(len,prompts)),wall_ms=(time.perf_counter()-start)*1000,records=records)

for pack in packs[:3]:run_pack(pack)
if args.profile_dir:
    session.post(args.base+'/start_profile',json=dict(output_dir=args.profile_dir,activities=['CPU','GPU'],with_stack=False,record_shapes=True),timeout=120).raise_for_status()
    try:results=[run_pack(packs[args.profile_pack])]
    finally:session.post(args.base+'/stop_profile',timeout=1200).raise_for_status()
else:
    results=[]
    journal=pathlib.Path(args.output).with_suffix('.jsonl')
    with journal.open('w') as file:
        for index,pack in enumerate(packs):
            result=run_pack(pack);result['pack_index']=index;results.append(result)
            file.write(json.dumps(result)+'\n');file.flush()
            if index%10==0:print('PACK',index,'/',len(packs),'BS',result['requests'],'TOKENS',result['tokens'],'MS',result['wall_ms'],flush=True)
values=[r['ttft_ms'] for p in results for r in p['records']]
wall=sum(p['wall_ms'] for p in results)
summary=dict(requests=len(values),packs=len(results),total_input_tokens=sum(p['tokens'] for p in results),mean_ttft_ms=statistics.mean(values),median_ttft_ms=statistics.median(values),p95_ttft_ms=sorted(values)[int((len(values)-1)*0.95)],total_burst_wall_ms=wall,requests_per_second=len(values)/(wall/1000))
pathlib.Path(args.output).write_text(json.dumps(dict(summary=summary,dataset_sha256=identity,packs=results),indent=2))
print('SUMMARY',json.dumps(summary),flush=True)
