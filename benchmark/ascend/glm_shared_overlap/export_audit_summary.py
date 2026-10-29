"""Create a text-free audit; detailed responses and configurations stay remote."""
import argparse
import csv
import hashlib
import json
import pathlib
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', type=pathlib.Path, required=True)
parser.add_argument('--repo', type=pathlib.Path, default='/sgl-workspace/sglang')
args = parser.parse_args()
root = args.results_dir
comparison = json.loads((root / 'final-gsm8k-comparison.json').read_text())
assert comparison['rescored_records'] == 2638
metric_keys = ['count', 'correct', 'accuracy', 'truncated', 'shots', 'temperature',
               'max_new_tokens', 'parallel']
summary = {key: comparison[key] for key in [
    'count', 'prompt_hashes_equal', 'accuracy_delta_percentage_points',
    'same_predicted_answer', 'same_full_text', 'baseline_correct_optimized_wrong',
    'baseline_wrong_optimized_correct', 'rescored_records']}
for variant in ['baseline', 'optimized']:
    summary[variant] = {key: comparison[variant][key] for key in metric_keys}
    assert all(isinstance(value, (int, float)) for value in summary[variant].values())
summary['detailed_records_remain_remote'] = True
(root / 'final-gsm8k-summary.json').write_text(json.dumps(summary, indent=2))
base = json.loads((root / 'baseline-final-gsm8k.json').read_text())
opt = json.loads((root / 'optimized-final-gsm8k.json').read_text())
aa = {r['index']: r for r in base['records']}
bb = {r['index']: r for r in opt['records']}
assert set(aa) == set(bb) == set(range(1319))
with (root / 'final-case-audit.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=[
        'case_index', 'prompt_equal', 'predicted_answer_equal', 'full_text_equal',
        'baseline_correct', 'optimized_correct', 'baseline_truncated', 'optimized_truncated'])
    writer.writeheader()
    for index in range(1319):
        a, b = aa[index], bb[index]
        assert all(isinstance(r[k], bool) for r in [a, b] for k in ['correct', 'truncated'])
        writer.writerow(dict(case_index=index,
                             prompt_equal=a['prompt_sha256'] == b['prompt_sha256'],
                             predicted_answer_equal=a['predicted'] == b['predicted'],
                             full_text_equal=a['text'] == b['text'],
                             baseline_correct=a['correct'], optimized_correct=b['correct'],
                             baseline_truncated=a['truncated'], optimized_truncated=b['truncated']))
configs = [json.loads((root / (p + '-final-quality-server.json')).read_text())
           for p in ['baseline', 'optimized']]
assert configs[0]['cmd'] == configs[1]['cmd']
envs = [{k:v for k,v in c['env'].items() if k != 'SGLANG_NPU_SHARED_EXPERT_OVERLAP'}
        for c in configs]
assert envs[0] == envs[1]
assert all(c['env']['ASCEND_RT_VISIBLE_DEVICES'] == '5' for c in configs)
original = (root / 'baseline-runtime.patch').read_bytes()
current = subprocess.check_output(['git', 'diff', '--binary'], cwd=args.repo)
assert current == original
component = json.loads((root / 'component-tuning.json').read_text())['results']
assert len(component) == 44 and all(r['bitwise_equal'] and r['max_abs'] == 0 for r in component)
text = (root / 'npu-tests.log').read_text()
unit, _ = json.JSONDecoder().raw_decode(text[text.index('{'):])
assert unit['status'] == 'passed' and all(r['bitwise_equal'] and r['max_abs'] == 0 for r in unit['cases'])
checksums = []
for name in ['baseline-final-gsm8k.json', 'optimized-final-gsm8k.json']:
    path = root / name
    with path.open('rb') as source:
        checksums.append(dict(role=name.removesuffix('.json'), bytes=path.stat().st_size,
                              sha256=hashlib.file_digest(source, 'sha256').hexdigest()))
audit = dict(quality=summary, matched_quality_configuration=True, physical_npu=5,
             component_cases=44, component_all_bitwise_equal=True,
             shared_projection_unit_cases=[{k:r[k] for k in ['tokens', 'bitwise_equal', 'max_abs']}
                                           for r in unit['cases']],
             exception_cleanup=unit['exception_cleanup'], decode_fallback=unit['decode_fallback'],
             original_runtime_preserved=True,
             original_runtime_patch_sha256=hashlib.sha256(original).hexdigest(),
             detailed_record_checksums=checksums,
             export_scope='Numbers, booleans and file digests only; no questions, responses, logs or launch configs.')
(root / 'final-audit-summary.json').write_text(json.dumps(audit, indent=2))
print(json.dumps(dict(summary='final-gsm8k-summary.json', cases=1319,
                      rescored_records=2638, text_exported=False), indent=2))
