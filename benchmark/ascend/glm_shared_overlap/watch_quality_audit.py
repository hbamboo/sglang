"""Wait for the existing quality controller, then independently audit both runs."""
import datetime
import json
import pathlib
import subprocess
import time
import traceback

root = pathlib.Path('/mnt/paas/hxdnpu/glm-shared-overlap-20261030')
controller = json.loads((root / 'final-validation-retry1-job.json').read_text())['pid']
proc = pathlib.Path('/proc', str(controller))
assert proc.exists() and 'run_final_validation.py' in proc.joinpath('cmdline').read_bytes().decode()
started = proc.joinpath('stat').read_text().split()[21]


def status(phase, **fields):
    data = dict(phase=phase, controller_pid=controller,
                updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields)
    temporary = root / 'quality-audit-state.tmp'
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(root / 'quality-audit-state.json')
    print(json.dumps(data), flush=True)


try:
    status('waiting-for-existing-quality-controller')
    while True:
        state = json.loads((root / 'validation-state.json').read_text())
        assert state['pid'] == controller, state
        if state.get('finished'):
            assert state['phase'] == 'final-evidence-ready', state
            break
        stat = proc.joinpath('stat')
        assert stat.exists(), f'Controller {controller} disappeared before completion'
        fields = stat.read_text().split()
        assert fields[2] != 'Z' and fields[21] == started, fields
        time.sleep(10)
    status('rescoring-all-2638-responses')
    with (root / 'final-gsm8k-audit.log').open('w') as log:
        subprocess.run(['python3', str(root / 'compare_gsm8k.py'),
                        '--results-dir', str(root), '--rescore'],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((root / 'final-gsm8k-comparison.json').read_text())
    assert result['rescored_records'] == 2638
    status('quality-audit-complete',
           baseline_accuracy=result['baseline']['accuracy'],
           optimized_accuracy=result['optimized']['accuracy'],
           delta_percentage_points=result['accuracy_delta_percentage_points'])
except BaseException:
    status('failed', error=traceback.format_exc())
    raise
