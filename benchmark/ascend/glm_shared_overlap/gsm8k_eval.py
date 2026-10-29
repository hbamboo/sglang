"""Paired full GSM8K test-set evaluation using local data and fixed prompts."""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import time

import pandas as pd
import requests
from evalscope.metrics.math.parser import extract_answer, math_equal, strip_answer_string
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--base', default='http://127.0.0.1:19255')
parser.add_argument('--output', required=True)
parser.add_argument('--limit', type=int, default=1319)
parser.add_argument('--parallel', type=int, default=16)
parser.add_argument('--max-new-tokens', type=int, default=8192)
args = parser.parse_args()
root = pathlib.Path('/mnt/paas/hmxnpu/gsm8k/main')
test = pd.read_parquet(root / 'test-00000-of-00001.parquet').head(args.limit)
train = pd.read_parquet(root / 'train-00000-of-00001.parquet').head(4)
tokenizer = AutoTokenizer.from_pretrained('/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8', trust_remote_code=True)
few_shot = []
for _, item in train.iterrows():
    reasoning, answer = item['answer'].rsplit('####', 1)
    few_shot.append(f"{item['question']}\n\nReasoning:\n{reasoning.strip()}\n\nANSWER: \\boxed{{{answer.strip()}}}")
examples = '\n\n'.join(few_shot)
started = time.monotonic()
path = pathlib.Path(args.output)
journal = path.with_suffix('.jsonl')
assert not journal.exists(), f'Refusing to overwrite existing evaluation {journal}'

def evaluate(pair):
    index, item = pair
    content = f'Here are some examples of how to solve similar problems:\n\n{examples}\n\n{item["question"]}\nPlease reason step by step, and put your final answer within \\boxed{{}}.'
    prompt = tokenizer.apply_chat_template([dict(role='user', content=content)], tokenize=False, add_generation_prompt=True)
    response = requests.post(args.base + '/generate', json=dict(text=prompt, sampling_params=dict(temperature=0, top_p=1, max_new_tokens=args.max_new_tokens)), timeout=1800)
    response.raise_for_status()
    result = response.json()
    text = result['text']
    target = item['answer'].rsplit('####', 1)[1].strip()
    predicted = extract_answer(text)
    correct = bool(math_equal(predicted, strip_answer_string(target)))
    finish = result['meta_info'].get('finish_reason', {})
    return dict(index=int(index), question=item['question'], target=target, predicted=predicted, correct=correct, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), text=text, meta_info=result['meta_info'], truncated=finish.get('type')=='length' if isinstance(finish,dict) else False)

records = []
with journal.open('w') as output, concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
    futures = [pool.submit(evaluate, item) for item in test.iterrows()]
    for future in concurrent.futures.as_completed(futures):
        record = future.result()
        records.append(record)
        output.write(json.dumps(record, ensure_ascii=False) + '\n')
        output.flush()
        if len(records) % 10 == 0 or len(records)==len(test):
            print('PROGRESS', len(records), '/', len(test), 'correct', sum(r['correct'] for r in records), 'seconds', round(time.monotonic()-started,1), flush=True)
summary = dict(count=len(records), correct=sum(r['correct'] for r in records), accuracy=sum(r['correct'] for r in records)/len(records), truncated=sum(r['truncated'] for r in records), seconds=time.monotonic()-started, shots=4, temperature=0, max_new_tokens=args.max_new_tokens, parallel=args.parallel, records=sorted(records, key=lambda r:r['index']))
path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print('SUMMARY', json.dumps({k:v for k,v in summary.items() if k!='records'}), flush=True)
