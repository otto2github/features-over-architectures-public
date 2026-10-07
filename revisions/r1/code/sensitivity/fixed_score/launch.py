#!/usr/bin/env python3
"""Start only this bounded CPU job; do not alter any existing experimental file."""
from __future__ import annotations
import fcntl, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
CYQ=Path('/path/to/private_workspace').resolve()
BASE=CYQ/'peerj_141707_v27/postfreeze_fixedscore_v1'
HERE=Path(__file__).resolve().parent
PY=CYQ/'peerj_141707_rerun_v1/.venv/bin/python'

def safe(p):
    rp=p.resolve()
    if not (rp==BASE.resolve() or BASE.resolve() in rp.parents):raise SystemExit('WRITE_OR_PACKAGE_OUTSIDE_ALLOWED_OUTPUT_ROOT')
    return p

if CYQ not in BASE.resolve().parents:raise SystemExit('SYMLINKED_OUTPUT_OUTSIDE_CYQ')
os.umask(0o077)
safe(HERE);safe(BASE);BASE.mkdir(parents=True,exist_ok=True)
if len(sys.argv)>1:raise SystemExit('No arguments: the frozen launcher uses two CPU workers.')
if not PY.is_file():raise SystemExit('PINNED_PYTHON_MISSING_NO_INSTALL_ATTEMPTED')
with safe(BASE/'LAUNCH.lock').open('a+') as guard:
    fcntl.flock(guard.fileno(),fcntl.LOCK_EX)
    running=safe(BASE/'RUNNING.json')
    if running.exists():
        j=json.loads(running.read_text());pid=int(j.get('pid',0))
        proc=Path(f'/proc/{pid}/cmdline')
        try:cmd=proc.read_bytes()
        except FileNotFoundError:cmd=b''
        if b'run_fixedscore.py' in cmd and str(HERE).encode() in cmd:
            print('ALREADY_RUNNING',pid,j.get('log'));raise SystemExit(0)
    latest=BASE/'LATEST_RETURN.json'
    if latest.is_file() and json.loads(latest.read_text()).get('status','').startswith('COMPLETE'):
        print('EXISTING_COMPLETE_RETURN_REFUSE_DUPLICATE',str(latest));raise SystemExit(0)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+f'_{os.getpid()}'
    attempt=safe(BASE/'attempts'/stamp);attempt.mkdir(parents=True,exist_ok=False)
    log=safe(attempt/'run.log');env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
               TMPDIR=str(attempt/'TMP'))
    (attempt/'TMP').mkdir()
    with log.open('xb') as stream:
        p=subprocess.Popen([str(PY),'-B','-u',str(HERE/'run_fixedscore.py'),'--run-dir',str(attempt),'--workers','2'],
                           cwd=str(attempt),stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                           start_new_session=True,env=env)
    info={'pid':p.pid,'attempt_dir':str(attempt),'log':str(log),'writes_only_under':str(BASE),
          'source_artifacts_read_only':True,'workers':2,'new_model_fits':0,'new_model_inferences':0}
    tmp=safe(BASE/'RUNNING.json.tmp');tmp.write_text(json.dumps(info,indent=2,sort_keys=True)+'\n');os.replace(tmp,running)
    print(json.dumps(info,indent=2,sort_keys=True))
