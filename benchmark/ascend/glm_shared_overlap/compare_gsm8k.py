"""Audit complete paired GSM8K runs, including prompts, answers and truncation."""
import argparse
import json
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', type=pathlib.Path, required=True)
parser.add_argument('--baseline', default='baseline-final-gsm8k')
parser.add_argument('--optimized', default='optimized-final-gsm8k')
args = parser.parse_args()
root = args.results_dir
base = json.loads((root / (args.baseline + '.json')).read_text(encoding='utf-8'))
opt = json.loads((root / (args.optimized + '.json')).read_text(encoding='utf-8'))
assert base['count'] == opt['count'] == 1319
for key in ['shots', 'temperature', 'max_new_tokens', 'parallel']:
    assert base[key] == opt[key], key
aa = {r['index']: r for r in base['records']}
bb = {r['index']: r for r in opt['records']}
assert set(aa) == set(bb) == set(range(1319))
differences = []
for index in aa:
    a, b = aa[index], bb[index]
    for key in ['question', 'target', 'prompt_sha256']:
        assert a[key] == b[key], (index, key)
    for r in [a, b]:
        assert not r['meta_info'].get('num_retractions', 0), index
        assert r['meta_info']['finish_reason']['type'] in ['stop', 'length'], (index, r['meta_info'])
    if any(a[k] != b[k] for k in ['predicted', 'correct', 'text', 'truncated']):
        differences.append(dict(index=index, target=a['target'], baseline=a['predicted'], optimized=b['predicted'],
                                baseline_correct=a['correct'], optimized_correct=b['correct'],
                                baseline_truncated=a['truncated'], optimized_truncated=b['truncated'],
                                text_equal=a['text'] == b['text']))
summary = dict(count=1319, prompt_hashes_equal=1319,
               baseline={k:v for k,v in base.items() if k != 'records'},
               optimized={k:v for k,v in opt.items() if k != 'records'},
               accuracy_delta_percentage_points=100*(opt['accuracy']-base['accuracy']),
               same_predicted_answer=sum(aa[i]['predicted']==bb[i]['predicted'] for i in aa),
               same_full_text=sum(aa[i]['text']==bb[i]['text'] for i in aa),
               baseline_correct_optimized_wrong=sum(aa[i]['correct'] and not bb[i]['correct'] for i in aa),
               baseline_wrong_optimized_correct=sum(not aa[i]['correct'] and bb[i]['correct'] for i in aa),
               differences=differences)
(root / 'final-gsm8k-comparison.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='differences'}, indent=2))
