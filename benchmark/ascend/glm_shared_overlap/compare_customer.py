"""Compare all aligned customer requests/packs without dropping outliers."""
import argparse
import csv
import json
import pathlib
import random
import statistics

parser=argparse.ArgumentParser()
parser.add_argument('--optimized',default='optimized-fresh-customer')
parser.add_argument('--results-dir',type=pathlib.Path)
args=parser.parse_args()
root=args.results_dir or pathlib.Path(__file__).parent/'results'
base=json.loads((root/'baseline-customer.json').read_text(encoding='utf-8'))
opt=json.loads((root/(args.optimized+'.json')).read_text(encoding='utf-8'))
assert base['dataset_sha256']==opt['dataset_sha256']
assert base['summary']['requests']==opt['summary']['requests']==2000
assert base['summary']['total_input_tokens']==opt['summary']['total_input_tokens']==7566887
records=[]
predictions=[]
for a,b in zip(base['packs'],opt['packs']):
    assert a['pack_index']==b['pack_index']
    assert a['tokens']==b['tokens'] and a['requests']==b['requests']
    assert a['tokens']<=32768
    first={r['dataset_index']:r for r in a['records']}
    second={r['dataset_index']:r for r in b['records']}
    assert first.keys()==second.keys()
    for index,x in first.items():
        y=second[index]
        assert x['input_tokens']==y['input_tokens']
        predictions.append(x['output_ids']==y['output_ids'])
    records.append(dict(pack=a['pack_index'],requests=a['requests'],tokens=a['tokens'],baseline_ms=a['wall_ms'],optimized_ms=b['wall_ms'],saved_ms=a['wall_ms']-b['wall_ms'],saved_pct=(1-b['wall_ms']/a['wall_ms'])*100))
assert len(records)==248
rng=random.Random(20261030)
means=[]
for _ in range(10000):
    sample=rng.choices(records,k=len(records))
    means.append(statistics.mean(r['saved_ms'] for r in sample))
means.sort()
summary=dict(baseline=base['summary'],optimized=opt['summary'],mean_ttft_reduction_pct=(1-opt['summary']['mean_ttft_ms']/base['summary']['mean_ttft_ms'])*100,median_ttft_reduction_pct=(1-opt['summary']['median_ttft_ms']/base['summary']['median_ttft_ms'])*100,p95_ttft_reduction_pct=(1-opt['summary']['p95_ttft_ms']/base['summary']['p95_ttft_ms'])*100,burst_wall_reduction_pct=(1-opt['summary']['total_burst_wall_ms']/base['summary']['total_burst_wall_ms'])*100,packs_improved=sum(r['saved_ms']>0 for r in records),paired_pack_saved_mean_ms=statistics.mean(r['saved_ms'] for r in records),paired_pack_saved_median_ms=statistics.median(r['saved_ms'] for r in records),paired_pack_saved_mean_bootstrap95_ms=[means[250],means[9749]],same_output_token_count=sum(predictions),output_token_count=len(predictions),outliers_excluded=0)
(root/(args.optimized+'-comparison.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8')
with (root/(args.optimized+'-paired-packs.csv')).open('w',newline='',encoding='utf-8-sig') as file:
    writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
print(json.dumps(summary,indent=2))
