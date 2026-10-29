"""Isolated matched allocator experiment; no automatic acceptance claim."""
import json
import pathlib
import traceback

import run_validation as control

OUT=control.OUT

def customer(name,profile=False):
    command=['python3',str(OUT/'customer_benchmark.py'),'--output',str(OUT/(name+'.json'))]
    if profile:command+=['--profile-dir',str(OUT/name),'--profile-repeat','5']
    control.child(name,command)

try:
    current=json.loads((OUT/'baseline-gsm8k-server.json').read_text())
    control.status('retained-baseline-stop')
    control.stop_server(current)
    template=json.loads((OUT/'baseline-server.json').read_text())
    template['env']['PYTORCH_NPU_ALLOC_CONF']='expandable_segments:True'
    baseline=control.start_server('baseline-expandable',False,template)
    customer('baseline-expandable-customer')
    customer('baseline-expandable-customer-profile',True)
    control.stop_server(baseline)
    optimized=control.start_server('optimized-expandable',True,template)
    customer('optimized-expandable-customer')
    customer('optimized-expandable-customer-profile',True)
    control.status('allocator-comparison-ready',server_pid=optimized['pid'],finished=True)
except BaseException:
    control.status('failed',error=traceback.format_exc(),finished=True)
    raise
