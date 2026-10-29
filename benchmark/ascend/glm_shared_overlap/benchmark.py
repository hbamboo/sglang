"""Exact 32768 input tokens / one output token, streaming TTFT benchmark."""
import argparse
import hashlib
import json
import pathlib
import statistics
import time

import requests
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--base', default='http://127.0.0.1:19255')
parser.add_argument('--output', required=True)
parser.add_argument('--warmup', type=int, default=3)
parser.add_argument('--repeat', type=int, default=15)
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained('/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8', trust_remote_code=True)
piece = tokenizer.encode('请阅读以下技术资料，最后概括要点。机器学习模型通过训练数据学习模式，推理系统关注准确率和响应时间。', add_special_tokens=False)
ids = (piece * (32768 // len(piece) + 1))[:32768]
identity = hashlib.sha256(json.dumps(ids).encode()).hexdigest()

def request_once():
    payload = dict(input_ids=ids, stream=True, sampling_params=dict(temperature=0, max_new_tokens=1, ignore_eos=True))
    start = time.perf_counter()
    first = None
    result = None
    with requests.post(args.base + '/generate', json=payload, stream=True, timeout=900) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b'data: '):
                continue
            raw = line[6:]
            if raw == b'[DONE]':
                break
            item = json.loads(raw)
            if item.get('meta_info', {}).get('completion_tokens', 0) > 0:
                if first is None:
                    first = (time.perf_counter() - start) * 1000
                result = item
    assert first is not None and result is not None
    meta = result['meta_info']
    assert meta['prompt_tokens'] == 32768, meta
    assert meta['completion_tokens'] == 1, meta
    assert meta.get('cached_tokens', 0) == 0, meta
    return dict(ttft_ms=first, total_ms=(time.perf_counter()-start)*1000, result=result)

warmup = []
for index in range(args.warmup):
    record = request_once()
    warmup.append(record)
    print('WARMUP', index, record['ttft_ms'], flush=True)
records = []
for index in range(args.repeat):
    record = request_once()
    records.append(record)
    print('MEASURE', index, record['ttft_ms'], flush=True)
    path = pathlib.Path(args.output)
    path.write_text(json.dumps(dict(input_tokens=32768, output_tokens=1, input_sha256=identity, warmup=warmup, records=records), indent=2, ensure_ascii=False))
values = [r['ttft_ms'] for r in records]
result = json.loads(path.read_text())
result['summary'] = dict(count=len(values), median_ms=statistics.median(values), mean_ms=statistics.mean(values), min_ms=min(values), max_ms=max(values), stdev_ms=statistics.stdev(values) if len(values)>1 else 0)
path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
print('SUMMARY', json.dumps(result['summary']), flush=True)
