"""Continue validation with the user-specified multi-request workload."""
import json

import run_validation as control

OUT=control.OUT

def customer(name,profile=False):
    command=['python3',str(OUT/'customer_benchmark.py'),'--output',str(OUT/(name+'.json'))]
    if profile:command+=['--profile-dir',str(OUT/name)]
    control.child(name,command)

try:
    optimized=json.loads((OUT/'optimized-server.json').read_text())
    customer('optimized-customer')
    customer('optimized-customer-profile',True)
    control.status('optimized-customer-stop')
    control.stop_server(optimized)
    template=json.loads((OUT/'baseline-server.json').read_text())
    baseline=control.start_server('baseline-customer',False,template)
    customer('baseline-customer')
    customer('baseline-customer-profile',True)
    control.child('baseline-gsm8k',['python3',str(OUT/'gsm8k_eval.py'),'--output',str(OUT/'baseline-gsm8k.json'),'--resume'])
    control.status('baseline-customer-stop')
    control.stop_server(baseline)
    optimized=control.start_server('optimized-gsm8k-resume',True,template)
    control.child('optimized-gsm8k-resume',['python3',str(OUT/'gsm8k_eval.py'),'--output',str(OUT/'optimized-gsm8k.json'),'--resume'])
    control.stop_server(optimized)
    control.status('complete',finished=True)
except BaseException:
    import traceback
    control.status('failed',error=traceback.format_exc(),finished=True)
    raise
