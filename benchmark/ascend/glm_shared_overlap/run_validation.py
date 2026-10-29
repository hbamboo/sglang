"""Run serial experiments on physical NPU 5; only stop owned server groups."""
import datetime
import json
import os
import pathlib
import signal
import subprocess
import time
import traceback

import requests

ROOT = pathlib.Path('/sgl-workspace/sglang')
OUT = pathlib.Path('/mnt/paas/hxdnpu/glm-shared-overlap-20261030')
BASE = 'http://127.0.0.1:19255'
state = dict(pid=os.getpid(), phase='created', completed=[])

def status(phase, **extra):
    state.update(phase=phase, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    temporary = OUT/'validation-state.tmp'
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(OUT/'validation-state.json')
    print(json.dumps(state), flush=True)

def child(name, cmd):
    log = (OUT/(name+'.log')).open('w')
    process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED='1'))
    status(name, child_pid=process.pid, child_cmd=cmd)
    code = process.wait()
    assert code == 0, f'{name} exited {code}; see {name}.log'
    state['completed'].append(name)

def benchmark(name):
    child(name, ['python3',str(OUT/'benchmark.py'),'--output',str(OUT/(name+'.json'))])

def profile(name):
    status(name)
    response = requests.post(BASE+'/start_profile', json=dict(output_dir=str(OUT/name), activities=['CPU','GPU'], with_stack=False, record_shapes=True), timeout=120)
    response.raise_for_status()
    try:
        child(name+'-request', ['python3',str(OUT/'benchmark.py'),'--warmup','0','--repeat','1','--output',str(OUT/(name+'-request.json'))])
    finally:
        requests.post(BASE+'/stop_profile', timeout=1200).raise_for_status()
    state['completed'].append(name)

def stop_server(info):
    pid=info['pid']
    path=pathlib.Path('/proc',str(pid),'cmdline')
    if path.exists():
        command=path.read_bytes().replace(b'\0',b' ').decode()
        assert 'sglang.launch_server' in command and '19255' in command and 'chatdpo-glm-flash-0727-mxfp8' in command, command
        assert os.getpgid(pid)==pid, 'Not an owned process group'
        os.killpg(pid,signal.SIGTERM)
        for _ in range(120):
            if not path.exists() or pathlib.Path('/proc',str(pid),'stat').read_text().split()[2]=='Z':
                break
            time.sleep(1)
        else:
            raise RuntimeError(f'Owned server {pid} did not stop')
    # Children use the same process group; startup on the same card follows
    # only after a short teardown interval, never a broad pkill.
    time.sleep(5)

def start_server(name, enabled, template):
    env=dict(os.environ, **template['env'])
    env.update(SGLANG_NPU_SHARED_EXPERT_OVERLAP=str(int(enabled)), PYTHONUNBUFFERED='1')
    process=subprocess.Popen(template['cmd'],cwd=ROOT,env=env,stdout=(OUT/(name+'-server.log')).open('w'),stderr=subprocess.STDOUT,start_new_session=True)
    info=dict(pid=process.pid,cmd=template['cmd'],env={k:env[k] for k in template['env']})
    (OUT/(name+'-server.json')).write_text(json.dumps(info,indent=2))
    status(name+'-startup',server_pid=process.pid)
    for _ in range(600):
        if process.poll() is not None:
            raise RuntimeError(f'{name} server exited {process.returncode}')
        try:
            result=requests.get(BASE+'/health',timeout=2)
            if result.status_code==200:
                return info
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f'{name} server readiness timeout')

try:
    baseline=json.loads((OUT/'baseline-server.json').read_text())
    profile('baseline-profile')
    status('baseline-stop')
    stop_server(baseline)
    optimized=start_server('optimized',True,baseline)
    benchmark('optimized-ttft')
    profile('optimized-profile')
    child('optimized-gsm8k',['python3',str(OUT/'gsm8k_eval.py'),'--output',str(OUT/'optimized-gsm8k.json')])
    status('optimized-stop')
    stop_server(optimized)
    baseline2=start_server('baseline-repeat',False,baseline)
    benchmark('baseline-repeat-ttft')
    child('baseline-gsm8k',['python3',str(OUT/'gsm8k_eval.py'),'--output',str(OUT/'baseline-gsm8k.json')])
    status('baseline-repeat-stop')
    stop_server(baseline2)
    status('complete', finished=True)
except BaseException:
    status('failed', error=traceback.format_exc(), finished=True)
    raise
