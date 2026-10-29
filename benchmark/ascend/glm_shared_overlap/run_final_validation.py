"""Reverse-order TTFT repeat, then paired full GSM8K on physical NPU 5."""
import argparse
import json
import pathlib
import traceback

import run_validation as control

parser = argparse.ArgumentParser()
parser.add_argument('--quality-only', action='store_true')
args = parser.parse_args()
OUT = control.OUT


def completed_customer(name):
    path = OUT / (name + '.json')
    assert not path.exists(), f'Refusing to replace {path}'
    control.child(name, ['python3', str(OUT / 'customer_benchmark.py'),
                         '--output', str(path)])
    assert json.loads(path.read_text())['summary']['requests'] == 2000


try:
    previous = json.loads((OUT / 'validation-state.json').read_text())
    assert previous.get('finished'), previous
    assert previous['phase'] in ['selective-comparison-ready', 'failed'], previous
    if previous.get('server_pid'):
        control.stop_server(dict(pid=previous['server_pid']))
    template = json.loads((OUT / 'baseline-server.json').read_text())
    template['env'].update(
        PYTORCH_NPU_ALLOC_CONF='expandable_segments:True',
        SGLANG_NPU_SHARED_EXPERT_OVERLAP_MIN_TOKENS='8192',
        SGLANG_NPU_SHARED_EXPERT_OVERLAP_PREQUANT='0',
        SGLANG_NPU_SHARED_EXPERT_OVERLAP_CUBE_CORES='-1',
        SGLANG_NPU_SHARED_EXPERT_OVERLAP_VECTOR_CORES='-1')
    if not args.quality_only:
        for name, enabled in [('optimized-final-repeat', True),
                              ('baseline-final-repeat', False)]:
            server = control.start_server(name, enabled, template)
            completed_customer(name + '-customer')
            control.status(name + '-stop')
            control.stop_server(server)
    quality = dict(template, cmd=list(template['cmd']), env=dict(template['env']))
    quality['cmd'].remove('--disable-cuda-graph')
    quality['cmd'] += ['--cuda-graph-backend-prefill', 'disabled',
                       '--cuda-graph-backend-decode', 'full',
                       '--cuda-graph-max-bs-decode', '32',
                       '--cuda-graph-bs-decode', '32']
    (OUT / 'final-quality-template.json').write_text(json.dumps(quality, indent=2))
    for name, enabled in [('baseline-final', False), ('optimized-final', True)]:
        server = control.start_server(name + '-quality', enabled, quality)
        output = OUT / (name + '-gsm8k.json')
        assert not output.with_suffix('.jsonl').exists(), output
        control.child(name + '-gsm8k', [
            'python3', str(OUT / 'gsm8k_eval.py'), '--parallel', '32',
            '--limit', '1319', '--max-new-tokens', '8192',
            '--output', str(output)])
        result = json.loads(output.read_text())
        assert result['count'] == 1319
        control.status(name + '-quality-stop')
        control.stop_server(server)
    control.status('final-evidence-ready', finished=True)
except BaseException:
    control.status('failed', error=traceback.format_exc(), finished=True)
    raise
