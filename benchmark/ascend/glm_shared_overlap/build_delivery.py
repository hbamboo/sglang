"""Build the final review bundle only after full quality and performance audits."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', type=pathlib.Path, required=True)
parser.add_argument('--repo', type=pathlib.Path, default='/sgl-workspace/sglang')
args = parser.parse_args()
root = args.results_dir
quality = json.loads((root / 'final-gsm8k-comparison.json').read_text())
assert quality['count'] == quality['prompt_hashes_equal'] == 1319
assert quality['rescored_records'] == 2638
for prefix in ['optimized-selective-customer', 'optimized-final-repeat-customer']:
    comparison = json.loads((root / (prefix + '-comparison.json')).read_text())
    assert comparison['output_token_count'] == comparison['same_output_token_count'] == 2000
    assert comparison['outliers_excluded'] == 0
    assert comparison['mean_ttft_reduction_pct'] > 0
subprocess.run(['git', 'diff', '--check',
                '0bcd822377da7b5718e674eaf9c870d349424dd1', 'HEAD'], cwd=args.repo, check=True)
code = args.repo / 'benchmark/ascend/glm_shared_overlap'
names = ['baseline-runtime.patch', 'runtime-provenance.json',
         'final-gsm8k-comparison.json', 'final-profiling-manifest.json', 'npu-tests.log',
         'final-quality-template.json']
for prefix in ['baseline-expandable', 'optimized-selective', 'baseline-final-repeat',
               'optimized-final-repeat']:
    names += [prefix + '-server.json', prefix + '-customer.json']
for prefix in ['optimized-selective-customer', 'optimized-final-repeat-customer']:
    names += [prefix + '-comparison.json', prefix + '-paired-packs.csv']
for prefix in ['baseline-final', 'optimized-final']:
    names += [prefix + '-quality-server.json', prefix + '-gsm8k.json',
              prefix + '-gsm8k.jsonl', prefix + '-gsm8k.log']
files = [root / name for name in names]
files += [p for p in code.iterdir() if p.is_file() and p.suffix in ['.py', '.md', '.sh']
          and p.name != 'run_fresh_customer_validation.py']
assert all(p.is_file() for p in files), [str(p) for p in files if not p.is_file()]
patch = subprocess.check_output(['git', 'format-patch', '--stdout', '--no-signature',
                                '0bcd822377da7b5718e674eaf9c870d349424dd1..HEAD'], cwd=args.repo)
manifest = dict(git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.repo, text=True).strip(),
                external_profile_archive=json.loads((root / 'final-profiling-manifest.json').read_text()), files=[])
target = root / 'final-validation-evidence.zip'
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
    for p in files:
        name = ('evidence/' if p.parent == root else 'reproduce/') + p.name
        with p.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        manifest['files'].append(dict(path=name, bytes=p.stat().st_size, sha256=digest))
        archive.write(p, name)
    archive.writestr('commits.patch', patch)
    archive.writestr('MANIFEST.json', json.dumps(manifest, indent=2))
with target.open('rb') as f:
    summary = dict(path=target.name, bytes=target.stat().st_size,
                   sha256=hashlib.file_digest(f, 'sha256').hexdigest(), git_head=manifest['git_head'])
(root / 'final-evidence-manifest.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
