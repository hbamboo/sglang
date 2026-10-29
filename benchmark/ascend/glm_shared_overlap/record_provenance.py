"""Record the exact runtime inputs without importing or initializing an NPU."""
import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', required=True, type=pathlib.Path)
parser.add_argument('--repo', default='/sgl-workspace/sglang', type=pathlib.Path)
args = parser.parse_args()


def git(*argv):
    return subprocess.check_output(['git', *argv], cwd=args.repo, text=True).strip()


def digest(path):
    with path.open('rb') as source:
        return dict(path=str(path), bytes=path.stat().st_size,
                    sha256=hashlib.file_digest(source, 'sha256').hexdigest())


versions = {}
for name in ['torch', 'torch-npu', 'sglang', 'sgl-kernel-npu', 'transformers', 'evalscope']:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
files = [pathlib.Path('/mnt/sfs_turbo/hxd/CHAT-mxfp8-customer-2000.jsonl'),
         pathlib.Path('/mnt/paas/hmxnpu/gsm8k/main/test-00000-of-00001.parquet'),
         pathlib.Path('/mnt/paas/hmxnpu/gsm8k/main/train-00000-of-00001.parquet'),
         args.results_dir / 'baseline-runtime.patch']
model = pathlib.Path('/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8')
files += [p for p in model.iterdir() if p.suffix == '.json']
dirty = git('diff', '--name-only').splitlines()
files += [args.repo / name for name in dirty if (args.repo / name).is_file()]
manifest = dict(python=sys.version, platform=platform.platform(), packages=versions,
                git_branch=git('rev-parse', '--abbrev-ref', 'HEAD'),
                git_head=git('rev-parse', 'HEAD'), dirty_files=dirty,
                checkpoint=str(model), files=[digest(p) for p in files])
original_patch = (args.results_dir / 'baseline-runtime.patch').read_bytes()
current_patch = subprocess.check_output(['git', 'diff', '--binary'], cwd=args.repo)
assert current_patch == original_patch, 'The pre-existing runtime patch has changed'
manifest['preexisting_runtime_preserved'] = True
manifest['preexisting_runtime_patch_sha256'] = hashlib.sha256(original_patch).hexdigest()
for directory in ['/usr/local/Ascend/ascend-toolkit/latest', '/usr/local/Ascend/cann/latest']:
    p = pathlib.Path(directory)
    if p.exists():
        manifest.setdefault('cann_locations', []).append(str(p.resolve()))
for p in pathlib.Path('/usr/local/Ascend').glob('*/version.cfg'):
    manifest.setdefault('cann_versions', {})[str(p)] = p.read_text(errors='replace')[:6000]
output = args.results_dir / 'runtime-provenance.json'
output.write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(output=str(output), head=manifest['git_head'],
                      packages=versions, hashed_files=len(files)), indent=2))
