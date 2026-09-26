"""Bounded read-only input access and allowlisted aggregate output."""
from __future__ import annotations
import hashlib, json, os, re, sqlite3, sys, time, zipfile
from pathlib import Path
from datetime import datetime
from urllib.parse import quote
import numpy as np
import pandas as pd

class AuditStop(RuntimeError):
    """An expected safety/contract stop; message is a non-sensitive code."""

def require(ok, code):
    if not ok:
        raise AuditStop(code)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read_json(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)

def json_clean(obj):
    if isinstance(obj, dict): return {str(k):json_clean(v) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)): return [json_clean(x) for x in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, (float,np.floating)): return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, Path): return str(obj)
    if obj is pd.NA or obj is pd.NaT: return None
    return obj

def write_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(json_clean(data),f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def company_codes(s):
    x=s.astype('string').str.strip().str.replace(r"^'",'',regex=True)
    x=x.str.replace(r'\.0$','',regex=True).str.replace(r'\.(SZ|SH|BJ)$','',regex=True)
    x=x.str.replace(r'^C:','',regex=True).str.zfill(6)
    require(x.notna().all() and x.str.fullmatch(r'\d{6}').all(),'INVALID_COMPANY_CODE_FORMAT')
    return x.astype(str)

def dates(s):
    """Explicit YYYYMMDD/ISO parsing, never integer nanoseconds."""
    x=s.astype('string').str.strip().str.replace(r'\.0$','',regex=True)
    a=pd.to_datetime(x,format='%Y%m%d',errors='coerce')
    b=pd.to_datetime(x.str.slice(0,10),format='%Y-%m-%d',errors='coerce')
    return a.fillna(b).dt.normalize()

def partition(year):
    y=int(year)
    if 2010<=y<=2018:return 'train'
    if 2019<=y<=2020:return 'validation'
    if 2021<=y<=2022:return 'test'
    return 'outside_primary'

def quantiles(s):
    a=pd.to_numeric(pd.Series(s),errors='coerce').dropna().to_numpy(float)
    if not len(a): return {'n':0}
    q=np.quantile(a,[0,.25,.5,.75,.9,.95,1])
    return dict(zip(['n','mean','min','q25','median','q75','q90','q95','max'],[len(a),float(a.mean()),*q]))

class Context:
    def __init__(self, args, protocol, package):
        self.args=args;self.p=protocol;self.package=Path(package)
        self.audit=Path(args.audit_root).expanduser().resolve()
        self.project=Path(args.project_root).expanduser().resolve()
        self.role=args.role
        base=Path(args.output_root).expanduser().resolve() if args.output_root else self.audit/'final_checks_v16'/args.role
        # New output only. Never put it under the data/experiment project.
        require(base!=self.project and self.project not in base.parents,'OUTPUT_WITHIN_SOURCE_PROJECT')
        base.mkdir(parents=True,exist_ok=True,mode=0o700)
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
        self.out=base/stamp;self.out.mkdir(mode=0o700)
        self.summary=self.out/'share';self.summary.mkdir(mode=0o700)
        self.private=self.out/'LOCAL_ONLY';self.private.mkdir(mode=0o700)
        self.inputs={};self.exports=[];self.status=[];self.paths={}
        self.start=datetime.now().astimezone().isoformat()
        self.message('Output directory: '+str(self.out))
    def message(self,s):
        print(s,flush=True)
    def use(self,path,role,expected=None):
        path=Path(path).expanduser()
        require(path.is_file() and not path.is_symlink(),'MISSING_OR_SYMLINK_INPUT:'+role)
        path=path.resolve()
        if path in self.inputs:
            rec=self.inputs[path]
            require(expected is None or rec['sha256']==expected,'FROZEN_HASH_MISMATCH:'+role)
            return path
        stat=path.stat();self.message('  read/hash: '+role)
        h=sha256(path)
        require(expected is None or h==expected,'FROZEN_HASH_MISMATCH:'+role)
        require((stat.st_size,stat.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns),'INPUT_CHANGED_WHILE_HASHING:'+role)
        self.inputs[path]={'role':role,'basename':path.name,'sha256':h,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'pinned_hash_match':expected is not None}
        self.paths[role]=path
        return path
    def find_pinned(self,name):
        expected=self.p['frozen_hashes'][name]
        cand=[self.project/'data'/name]
        if name=='fraud_labels_v1_0.parquet':
            cand+=sorted((self.audit/'label_v1_stageF').glob('stageF_*/'+name))
        elif name=='node_id_index_label_v1_0_fiverel_derived.parquet':
            cand+=sorted((self.audit/'label_v1_stageG4').glob('stageG4_*/'+name))
        elif name=='features_v1_1.parquet':
            cand+=[self.project/'data/processed/features'/name]
        else: cand+=[self.project/'data/processed/kg'/name]
        existing=[x for x in cand if x.is_file() and not x.is_symlink()]
        require(existing,'FROZEN_INPUT_NOT_FOUND:'+name)
        # The authoritative direct path may not silently fall back after a hash mismatch.
        for x in existing:
            if x==self.project/'data'/name or x==self.project/'data/processed/kg'/name or x==self.project/'data/processed/features'/name:
                return self.use(x,name,expected)
        for x in existing:
            if sha256(x)==expected:return self.use(x,name,expected)
        raise AuditStop('NO_HASH_MATCHING_FROZEN_INPUT:'+name)
    def export_json(self,name,obj):
        require(Path(name).name==name,'INVALID_EXPORT_NAME')
        p=self.summary/name;write_json(p,obj);self.exports.append(p)
    def export_csv(self,name,rows):
        require(Path(name).name==name,'INVALID_EXPORT_NAME')
        p=self.summary/name;df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
        # Never allow an identifier-bearing column into the share ZIP.
        forbidden={'company_code','firm_id','node_id','node_idx','src_idx','dst_idx','row_ref','document_number','violation_id','score','y_true','annual_report_public_anchor','declare_date','file_name','canonical_name'}
        require(not (set(df.columns)&forbidden),'PRIVATE_COLUMNS_IN_EXPORT:'+name)
        df.to_csv(p,index=False,encoding='utf-8-sig');self.exports.append(p)
    def record(self,module,state,**extra):
        self.status.append({'module':module,'state':state,**json_clean(extra)})
        self.message(f'[{module}] {state}')
    def run(self,name,fn):
        try:
            fn()
        except AuditStop as e:
            self.record(name,'BLOCKED',reason=str(e))
        except Exception as e:
            import traceback
            # Traceback with potential private source content stays LOCAL_ONLY.
            p=self.private/(name+'_traceback.txt')
            p.write_text(traceback.format_exc(),encoding='utf-8')
            self.record(name,'ERROR',reason='UNEXPECTED_'+type(e).__name__,local_traceback=p.name)
    def finish(self):
        changes=[]
        for p,rec in self.inputs.items():
            try:
                s=p.stat()
                if (s.st_size,s.st_mtime_ns)!=(rec['bytes'],rec['mtime_ns']):changes.append(rec['role'])
            except OSError:
                changes.append(rec['role']+':MISSING_OR_UNREADABLE_AT_FINISH')
        if changes:self.record('input_stability','INVALID',roles=changes)
        else:self.record('input_stability','PASS',method='size_and_mtime_after_full_sha256_at_first_read',scope='no concurrent writer proven only to recorded checks')
        state='INVALID' if changes else ('PARTIAL' if any(s['state'] in ('BLOCKED','ERROR','PARTIAL') for s in self.status) else 'COMPLETE')
        self.export_json('RUN_STATUS.json',{'version':self.p['tool_version'],'baseline':'v16','role':self.role,'state':state,'started':self.start,'finished':datetime.now().astimezone().isoformat(),'training_executed':False,'paper_files_modified':False,'network_requests':False,'source_writes_requested':False,'checks':self.status})
        self.export_json('INPUT_FINGERPRINTS.json',[rec for rec in self.inputs.values()])
        write_json(self.private/'EXACT_INPUT_PATHS.json',{rec['role']:str(p) for p,rec in self.inputs.items()})
        lines=['# PeerJ 141707 v16 一次性核验结果',f'设备：{self.role}；状态：{state}','', '本包仅含聚合结果和核验状态；不含逐公司预测、个人节点、案件号或底层表。','本次没有训练模型、改标签文件、改论文或上传网络。','缺失/歧义使用 BLOCKED 或 PARTIAL，不能当作零或已排除问题。','PASS/COMPLETE 仅表示对应程序检查执行完毕，不表示没有时点问题或已达到发表标准。','', '| 模块 | 状态 | 说明 |','|---|---|---|']
        for s in self.status:lines.append('| '+s['module']+' | '+s['state']+' | '+str(s.get('reason',s.get('note','')))+' |')
        p=self.summary/'SUMMARY_先读.md';p.write_text('\n'.join(lines)+'\n',encoding='utf-8');self.exports.append(p)
        # Explicit list of files created by exporters, not recursive source globbing.
        dest=self.out/f'peerj_final_check_{self.role}_RESULTS.zip'
        man={p.name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p in self.exports}
        with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in self.exports:z.write(p,p.name)
            z.writestr('SHA256_MANIFEST.json',json.dumps(man,indent=2)+'\n')
        self.message('\nUPLOAD THIS ZIP: '+str(dest))
        self.message('Do not upload LOCAL_ONLY. State: '+state)
        return dest

def load_labels(ctx):
    if hasattr(ctx,'labels'):return ctx.labels
    p=ctx.find_pinned('fraud_labels_v1_0.parquet')
    f=pd.read_parquet(p);label=ctx.p['primary_label']
    need=['company_code','fiscal_year',label,'annual_report_public_anchor','freeze_status']
    require(set(need)<=set(f.columns),'FROZEN_LABEL_SCHEMA')
    f=f.copy();f['company_code']=company_codes(f.company_code);f['fiscal_year']=f.fiscal_year.astype(int)
    require(not f.duplicated(['company_code','fiscal_year']).any(),'DUPLICATE_LABEL_KEYS')
    require(len(f)==ctx.p['expected_panel_rows'],'LABEL_PANEL_COUNT')
    f['partition']=f.fiscal_year.map(partition);f['target']=pd.to_numeric(f[label],errors='coerce')
    require(f.target.dropna().isin([0,1]).all(),'FROZEN_LABEL_VALUES')
    f['anchor']=dates(f.annual_report_public_anchor)
    for part,(n,pos) in ctx.p['expected_counts'].items():
        a=f[(f.partition==part)&f.target.notna()]
        require(len(a)==n and a.target.sum()==pos,'FROZEN_LABEL_COUNT:'+part)
    elig=f.target.notna();fu=(pd.Timestamp(ctx.p['cutoff'])-f.anchor).dt.days
    require(f.loc[elig,'anchor'].notna().all() and (fu[elig]>=1095).all(),'FROZEN_LABEL_MATURITY')
    require(not (elig & (f.partition=='outside_primary')).any(),'ELIGIBLE_OUTSIDE_PRIMARY_YEARS')
    f['label_group']=f.target.map({0.0:'negative',1.0:'positive'}).fillna('null')
    ctx.labels=f
    return f

def load_features(ctx):
    if hasattr(ctx,'features'):return ctx.features
    p=ctx.find_pinned('node_features_v1_1.parquet');f=pd.read_parquet(p)
    require({'firm_id','year','node_id'}<=set(f.columns),'NODE_FEATURE_SCHEMA')
    f=f.copy();f['company_code']=company_codes(f.firm_id);f['fiscal_year']=f.year.astype(int)
    require(len(f)==ctx.p['expected_panel_rows'] and not f.duplicated(['company_code','fiscal_year']).any(),'NODE_FEATURE_KEYS')
    ctx.features=f
    return f

def readonly_sqlite(path):
    """Use immutable read-only handle only for a quiescent DB; never ignore a live WAL."""
    p=Path(path).resolve()
    require(p.is_file(),'SQLITE_NOT_FOUND')
    for suffix in ('-wal','-journal'):
        side=Path(str(p)+suffix)
        require(not side.exists() or side.stat().st_size==0,'SQLITE_ACTIVE_SIDECAR_COPY_A_QUIESCENT_SNAPSHOT_FIRST')
    uri='file:'+quote(str(p),safe='/')+'?mode=ro&immutable=1'
    con=sqlite3.connect(uri,uri=True,timeout=10)
    con.execute('PRAGMA query_only=ON')
    return con
