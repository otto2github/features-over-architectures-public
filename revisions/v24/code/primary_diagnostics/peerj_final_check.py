#!/usr/bin/env python3
"""PeerJ 141707 v16: first-stage read-only audit/postprocessing, never training."""
from __future__ import annotations
import argparse, importlib.metadata, json, os, sys
from pathlib import Path

# Avoid writing __pycache__ inside source trees and prevent accidental CUDA use.
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')

from auditlib.common import Context,read_json,require,sha256
from auditlib.endpoints import cohort_summary,endpoint_audit
from auditlib.financials import financial_audit
from auditlib.graphs import graph_audit
from auditlib.predictions import prediction_audit,history_baseline

def main():
    package=Path(__file__).resolve().parent
    p=read_json(package/'config/protocol.json')
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--role',required=True,choices=['mac','server'])
    ap.add_argument('--project-root',help='Mac project snapshot or server rerun project; default is role-specific existing path')
    ap.add_argument('--audit-root',default=p['audit_default'],help='Existing local audit directory containing Stage A/E/F/G4 outputs')
    ap.add_argument('--reverse-root',help='Exact direction_sensitivity_v13 experiment directory on server')
    ap.add_argument('--financial-db',help='Explicit path to quiescent financials.db, read-only')
    ap.add_argument('--output-root',help='NEW outputs only, must be outside the source project')
    ap.add_argument('--bootstrap',type=int,default=p['bootstrap_replicates'])
    ap.add_argument('--workers',type=int,default=min(8,max(1,(os.cpu_count() or 2)//2)))
    a=ap.parse_args()
    if a.project_root is None:a.project_root=p[a.role+'_project_default']
    if a.bootstrap<100:ap.error('--bootstrap must be at least 100 for real-data runs')
    if not 1<=a.workers<=32:ap.error('--workers must be between 1 and 32')
    # Verify every shipped tool/reference byte before inspecting private data.
    manifest=read_json(package/'TOOL_SHA256.json')
    for rel,h in manifest.items():
        q=package/rel
        require(q.is_file() and not q.is_symlink() and sha256(q)==h,'TOOL_INTEGRITY_FAILED:'+rel)
    os.umask(0o077)
    ctx=Context(a,p,package)
    versions={}
    for name in ['numpy','pandas','pyarrow','scikit-learn']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    ctx.export_json('ENVIRONMENT.json',{'python':sys.version.split()[0],'packages':versions,'tool_version':p['tool_version'],'role':a.role,'bootstrap':a.bootstrap,'workers':a.workers,'CUDA_VISIBLE_DEVICES':'','training_executed':False,'baseline':'v16'})
    ctx.export_json('BASELINE_IDENTITY.json',read_json(package/'reference/BASELINE_IDENTITY.json'))
    ctx.export_json('TOOL_IDENTITY.json',{'tool_version':p['tool_version'],'tool_manifest_sha256':sha256(package/'TOOL_SHA256.json'),'payload_entries_verified':len(manifest),'startup_real_parquet_selftests':os.environ.get('PEERJ_STARTUP_SELFTEST','not_recorded_direct_python_entry')})
    ctx.run('frozen_cohort',lambda:cohort_summary(ctx))
    if a.role=='mac':
        ctx.run('endpoint_timing',lambda:endpoint_audit(ctx))
        ctx.run('financial_vintage',lambda:financial_audit(ctx))
    else:
        ctx.run('saved_prediction_postprocessing',lambda:prediction_audit(ctx))
        ctx.run('graph_hubs_and_coverage',lambda:graph_audit(ctx))
        ctx.run('history_no_propagation',lambda:history_baseline(ctx))
    ctx.finish()

if __name__=='__main__':
    main()
