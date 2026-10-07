#!/usr/bin/env python3
from __future__ import annotations
import json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=Path('/path/to/private_workspace/peerj_141707_v28_b0gnn_extension/fixedscore_completion_v1')
PY=Path('/path/to/private_workspace/peerj_141707_rerun_v1/.venv/bin/python')

def main():
    from extension_fixedscore import safe_output,verify_package,writej
    safe_output(BASE);safe_output(HERE);verify_package()
    if (BASE/'V28_FIXED_SCORE_RETURN.zip').exists():raise SystemExit('STOP_EXISTING_RETURN_PRESERVED: inspect the existing result before another attempt.')
    BASE.mkdir(parents=True,exist_ok=True)
    lock=BASE/'ACTIVE_LAUNCH.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:raise SystemExit('STOP_LAUNCH_LOCK_EXISTS: no automatic relaunch.')
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid())+'\n')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+str(os.getpid())
    attempt=BASE/'attempts'/stamp;attempt.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update({'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'})
    for key,dirname in [('HOME','home'),('TMPDIR','tmp'),('XDG_CACHE_HOME','cache')]:
        p=attempt/dirname;p.mkdir();env[key]=str(p)
    log=attempt/'console.log'
    with log.open('xb') as out:
        p=subprocess.Popen([str(PY),'-B',str(HERE/'extension_fixedscore.py'),'--attempt',str(attempt),'--workers','2'],cwd=str(attempt),stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    info=dict(pid=p.pid,attempt=str(attempt),log=str(log),started_at_utc=datetime.now(timezone.utc).isoformat(),new_fits=0,new_inference=0)
    writej(BASE/'RUNNING.json',info);print(json.dumps(info,indent=2))
    print('The launcher does not train or infer any model. Preserve prior outputs on error.')
if __name__=='__main__':main()
