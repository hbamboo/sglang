"""Evaluate the evidence-based small-tail selector after allocator A/B."""
import json
import pathlib
import time
import traceback

import run_validation as control

OUT=control.OUT

def live_non_zombie(pid):
    p=pathlib.Path('/proc',str(pid),'stat')
    return p.exists() and p.read_text().split()[2]!='Z'

def customer(name,profile=False):
    cmd=['python3',str(OUT/'customer_benchmark.py'),'--output',str(OUT/(name+'.json'))]
    if profile:cmd+=['--profile-dir',str(OUT/name),'--profile-repeat','5']
    control.child(name,cmd)

try:
    predecessor=json.loads((OUT/'allocator-validation-job.json').read_text())['pid']
    print('WAITING_FOR_ALLOCATOR_JOB',predecessor,flush=True)
    while live_non_zombie(predecessor):time.sleep(5)
    state=json.loads((OUT/'validation-state.json').read_text())
    assert state['phase']=='allocator-comparison-ready',state
    for prefix in ['baseline-expandable-customer','optimized-expandable-customer']:
        result=json.loads((OUT/(prefix+'.json')).read_text())
        assert result['summary']['requests']==2000
    previous=json.loads((OUT/'optimized-expandable-server.json').read_text())
    control.status('selective-previous-stop')
    control.stop_server(previous)
    template=json.loads((OUT/'baseline-server.json').read_text())
    template['env'].update(PYTORCH_NPU_ALLOC_CONF='expandable_segments:True',SGLANG_NPU_SHARED_EXPERT_OVERLAP_MIN_TOKENS='8192')
    server=control.start_server('optimized-selective',True,template)
    customer('optimized-selective-customer')
    customer('optimized-selective-customer-profile',True)
    control.status('selective-comparison-ready',server_pid=server['pid'],finished=True)
except BaseException:
    control.status('failed',error=traceback.format_exc(),finished=True)
    raise
