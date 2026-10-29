"""Bundle the matched final customer traces, CSVs, metadata and checksums."""
import argparse
import hashlib
import json
import pathlib
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', type=pathlib.Path, required=True)
args = parser.parse_args()
root = args.results_dir
name = 'final-customer-profiling.zip'
assert not (root / name).exists(), root / name
prefixes = ['baseline-expandable-customer', 'optimized-selective-customer']
files = []
for prefix in prefixes:
    result = root / (prefix + '.json')
    assert json.loads(result.read_text())['summary']['requests'] == 2000
    files += [result, root / (prefix + '-profile.json')]
    profile = root / (prefix + '-profile')
    for p in profile.rglob('*'):
        if p.is_file() and (p.parent.name == 'ASCEND_PROFILER_OUTPUT' or p.name.startswith('profiler_info')):
            files.append(p)
manifest = dict(profiles=prefixes, capture_repetitions=5, analysis_discard_first=2,
                note='Controlled customer bursts; profiling is separate from timed TTFT runs.', files=[])
for p in files:
    manifest['files'].append(dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size,
                                  sha256=hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()))
with zipfile.ZipFile(root / name, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
    for p in files:
        archive.write(p, p.relative_to(root))
    archive.writestr('MANIFEST.json', json.dumps(manifest, indent=2))
summary = dict(path=name, bytes=(root / name).stat().st_size,
               sha256=hashlib.file_digest((root / name).open('rb'), 'sha256').hexdigest(), files=len(files)+1)
(root / 'final-profiling-manifest.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
