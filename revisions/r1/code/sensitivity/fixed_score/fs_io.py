"""Strict immutable input binding and controlled output for V27 fixed scores."""
from __future__ import annotations
import hashlib, json, os, re
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from fs_math import ContractError, require, SEEDS, condition_name

PRIVATE_ROOT=Path('/path/to/private_workspace')
ROOT=PRIVATE_ROOT/'peerj_141707_v27'
WRITE_ROOT=ROOT/'postfreeze_fixedscore_v1'
KEYS=['company_code','fiscal_year']
POPS=['full_test','D_frozen','O_frozen','S_frozen','J_frozen']
POP_COUNTS={'full_test':(8435,20),'D_frozen':(8422,7),'O_frozen':(8428,13),'S_frozen':(7757,6),'J_frozen':(7753,2)}
EXPECTED={
 'bindings/RUN_BINDINGS_v2.json':'cebbfada1bf30ace4f2c95d8fc3b2e60fe9f084294026ea0071213b88a7f4325',
 'bindings/RUN_BINDINGS_v2_DATA_STAGE.json':'71ebdb3ec4521b9c09773ce2a8a7cec64e240b0fb092a113a8281a49c745aaa6',
 'bindings/P2A_REFERENCE_BINDING.json':'8cb66579086c6211beef55d8ecca65846f044cfd4787083f02926ef730ed5e81',
 'bindings/P2A_PRIVATE_REFERENCE_PATHS.json':'de484348197dd4bd95e87ae18de2b7f0579b5a06f8772ebfa4850dea89731492',
 'bindings/sensitivity_run_matrix.json':'4d78371987154fa5aee660479c447e30f56608d28616359f04d8c472759f8923',
 'stage_t/ALL_105_LOCKED.json':'72dbfdeeadad9742a3c80a15e1bb96abebe88be67461b53539932c195dcf805b',
 'stage_e_freeze_v1/SENSITIVITY_ARTIFACT_IDENTITIES.json':'4a0086b676a5df8c7a7c085f44dcfd3722ae4d5f8b3a081bae841d907de95c06',
 'stage_e_freeze_v1/SENSITIVITY_RESULTS.json':'1252202dfcf2261ee2227e2ee8549d525f8b07343cf0cc589bf4a7d1997f49f8',
 'inputs/base/fraud_labels_v1_0.parquet':'1d50378138dfc65def76555c0a48aa9cea2656af4d456fe5178d9cf7eae31836',
 'inputs/case/evaluation_masks_v27.parquet':'f53fa5b7f0ee845f6075bc2329d5ef6ee9d9429d38a942a71aa228f550009810',
 'inputs/case/positive_document_key_hash_incidence_v27.parquet':'9ce1c6a3881bd11716922e89d39a33d6de9cbada5c59fc83fb3a1a9144e8acae',
 'inputs/case/document_caseblock_masks_v27.parquet':'c88d4845a770e5e85b87cd9b44e2db2ab001a6574c7de48a454e5442f4508fda',
 'inputs/financial/financial_status_masks_v27_frozen.parquet':'815df9f31c928810d124238d57ee5b4c0e5170a62238411455fa94952bf9ddb7',
 'code/sensitivity_runner.py':'9e8efa8caf9145b748c01cb6fe86868136afd9f4cd33907fb886668eb854c5b2',
 'code/sensitivity_orchestrator.py':'35d8171606726e3f9386f66b61370cb7e5810f206fba7f057cf81444007f1395',
 'code/sensitivity_lock_training.py':'6df5d26e595e699a1de4cd144320a3f44d817fd25abfeb18fe19b69363273fb1',
}


def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def jsonable(x):
    """Explicitly prevent row-level DataFrame/array export and invalid JSON."""
    if isinstance(x,np.generic): return jsonable(x.item())
    if isinstance(x,Path): return str(x)
    if isinstance(x,dict): return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [jsonable(v) for v in x]
    if x is None or isinstance(x,(str,bool,int)): return x
    if isinstance(x,float):
        require(np.isfinite(x),'NONFINITE_JSON_VALUE')
        return x
    raise ContractError('NONEXPORTABLE_RUNTIME_OBJECT',type(x).__name__)

class Output:
    def __init__(self,root,allowed=WRITE_ROOT):
        self.root=Path(root);self.allowed=Path(allowed).resolve()
        if Path(allowed)==WRITE_ROOT:
            require(PRIVATE_ROOT.resolve() in self.allowed.parents, 'SYMLINKED_OUTPUT_OUTSIDE_PRIVATE_ROOT')
        self.check(self.root)
        self.root.mkdir(parents=True,exist_ok=True)
    def check(self,path):
        resolved=Path(path).resolve()
        require(resolved==self.allowed or self.allowed in resolved.parents,'WRITE_SCOPE_VIOLATION')
        return Path(path)
    def json(self,relative,obj):
        p=self.check(self.root/relative);p.parent.mkdir(parents=True,exist_ok=True)
        text=json.dumps(jsonable(obj),indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n'
        tmp=p.with_name(p.name+f'.tmp.{os.getpid()}');self.check(tmp)
        with tmp.open('x',encoding='utf-8') as f:f.write(text)
        os.replace(tmp,p);return p
    def csv(self,relative,rows):
        p=self.check(self.root/relative);p.parent.mkdir(parents=True,exist_ok=True)
        # Only aggregate records prepared by callers; never arbitrary input tables.
        import csv
        rows=[jsonable(r) for r in rows]
        keys=list(dict.fromkeys(k for r in rows for k in r))
        tmp=p.with_name(p.name+f'.tmp.{os.getpid()}');self.check(tmp)
        with tmp.open('x',newline='',encoding='utf-8') as f:
            if keys:
                w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
        os.replace(tmp,p);return p

class Ledger:
    def __init__(self,read_roots):
        self.roots=[Path(p).resolve() for p in read_roots];self.entries={}
    def verify(self,p,expected,size=None,role='input'):
        p=Path(p);resolved=p.resolve()
        require(any(resolved==r or r in resolved.parents for r in self.roots),'READ_SCOPE_VIOLATION',role)
        require(p.is_file(),'MISSING_BOUND_INPUT',role)
        st=p.stat()
        require(size is None or st.st_size==int(size),'INPUT_SIZE_MISMATCH',role)
        previous=self.entries.get(str(p))
        signature=(st.st_size,st.st_mtime_ns,st.st_ino)
        if previous and previous['signature']==signature:
            got=previous['sha256']
        else:
            got=sha256(p)
            require((p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino)==signature,'INPUT_CHANGED_DURING_HASH',role)
        require(expected is None or got==expected,'INPUT_HASH_MISMATCH',role)
        self.entries[str(p)]=dict(role=role,sha256=got,size_bytes=st.st_size,signature=signature)
        return p
    def audit_again(self):
        for p,rec in self.entries.items():
            st=Path(p).stat()
            require((st.st_size,st.st_mtime_ns,st.st_ino)==rec['signature'],'INPUT_METADATA_CHANGED',rec['role'])
            require(sha256(Path(p))==rec['sha256'],'INPUT_CONTENT_CHANGED',rec['role'])
        return {'files_rehashed':len(self.entries),'all_unchanged':True}
    def public_rows(self):
        return [{'role':r['role'],'sha256':r['sha256'],'size_bytes':r['size_bytes']} for r in self.entries.values()]

def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def strict_keys(df):
    require(set(KEYS)<=set(df.columns),'KEY_COLUMNS_MISSING')
    d=df.copy()
    s=d.company_code.astype('string').str.strip()
    s=s.str.replace(r"^'",'',regex=True).str.replace(r'^C:','',regex=True)
    s=s.str.replace(r'\.0$','',regex=True).str.replace(r'\.(SZ|SH|BJ)$','',regex=True).str.zfill(6)
    require(s.notna().all() and s.str.fullmatch(r'\d{6}').all(),'COMPANY_CODE_DOMAIN')
    yr=pd.to_numeric(d.fiscal_year,errors='coerce')
    require(yr.notna().all() and np.isfinite(yr.to_numpy(float)).all(),'YEAR_NONFINITE')
    require(np.equal(yr.to_numpy(float),np.floor(yr.to_numpy(float))).all(),'FRACTIONAL_YEAR')
    d['company_code']=s.astype(str);d['fiscal_year']=yr.astype(np.int64)
    return d

def one_to_one(base,other,cols,role):
    require(not other.duplicated(KEYS).any(),'DUPLICATE_JOIN_KEYS',role)
    d=base[KEYS].merge(other[KEYS+cols],how='left',on=KEYS,sort=False,validate='one_to_one',indicator=True)
    require((d['_merge']=='both').all(),'MISSING_JOIN_ROWS',role)
    require(d[KEYS].reset_index(drop=True).equals(base[KEYS].reset_index(drop=True)),'JOIN_LEFT_ORDER',role)
    return d.drop(columns='_merge')

def binary_array(s,nullable=False):
    v=pd.to_numeric(s,errors='coerce')
    # Coercion is permitted only for actual missing input values, never bad text.
    require(not (v.isna() & s.notna()).any(),'INVALID_BINARY_TEXT')
    require(v.dropna().isin([0,1]).all(),'BINARY_LABEL_DOMAIN')
    require(nullable or v.notna().all(),'MISSING_BINARY_VALUE')
    return v.to_numpy(dtype=float,na_value=np.nan)

def bool_array(s):
    require(s.notna().all(),'MISSING_MASK_VALUE')
    require(s.isin([False,True,0,1]).all(),'NONBINARY_MASK_VALUE')
    return s.to_numpy(dtype=bool)

@dataclass
class Fit:
    run_id:str
    condition:str
    seed:int
    family:str
    model:str
    modality:str
    world:int|None
    score:np.ndarray
    threshold:float
    stored:dict

@dataclass
class Bundle:
    canonical:pd.DataFrame
    y:np.ndarray
    masks:dict
    flags:dict
    incidence:pd.DataFrame
    fits:list
    final_binding:dict
    data_binding:dict
    ledger:Ledger
    run_matrix:list
    first_fit_binding_hash:str


def prediction(path,canonical):
    d=strict_keys(pd.read_parquet(path,columns=KEYS+['y_true','score']))
    require(len(d)==len(canonical) and not d.duplicated(KEYS).any(),'PREDICTION_KEY_COUNT')
    yy=binary_array(d.y_true)
    require(np.isfinite(yy).all(),'PREDICTION_LABEL_NA')
    d['y_true']=yy.astype(int)
    d=d.sort_values(KEYS).reset_index(drop=True)
    require(d[KEYS].equals(canonical[KEYS]),'PREDICTION_KEY_IDENTITY')
    require(np.array_equal(d.y_true.to_numpy(int),canonical.target.to_numpy(int)),'PREDICTION_LABEL_IDENTITY')
    ss=pd.to_numeric(d.score,errors='coerce').to_numpy(dtype=float,na_value=np.nan)
    require(np.isfinite(ss).all() and (ss>=0).all() and (ss<=1).all(),'PREDICTION_SCORE_DOMAIN')
    return ss


def load_bundle(package,progress=print):
    package=Path(package)
    ledger=Ledger([ROOT,PRIVATE_ROOT/'peerj_141707_rerun_v1',PRIVATE_ROOT/'_home_user_peerj_moved_20260927',package])
    for rel,h in EXPECTED.items(): ledger.verify(ROOT/rel,h,role=rel)
    final=read_json(ROOT/'bindings/RUN_BINDINGS_v2.json')
    data=read_json(ROOT/'bindings/RUN_BINDINGS_v2_DATA_STAGE.json')
    matrix=read_json(ROOT/'bindings/sensitivity_run_matrix.json')
    require(matrix==read_json(package/'reference/RUN_MATRIX_v2.json'),'MATRIX_PACKAGE_MISMATCH')
    require(len(matrix)==105 and len({r['run_id'] for r in matrix})==105,'RUN_MATRIX_IDENTITY')
    labels=strict_keys(pd.read_parquet(ROOT/'inputs/base/fraud_labels_v1_0.parquet',columns=KEYS+['split_v1','label_v1_strict_ab_primary']))
    require(len(labels)==51675 and not labels.duplicated(KEYS).any(),'LABEL_KEYS')
    labels['target']=binary_array(labels.label_v1_strict_ab_primary,nullable=True)
    canonical=labels[(labels.split_v1=='test_2021_2022') & labels.target.notna()][KEYS+['target']].sort_values(KEYS).reset_index(drop=True)
    y=canonical.target.to_numpy(int)
    require((len(y),int(y.sum()))==(8435,20),'TEST_COUNTS')
    mm=strict_keys(pd.read_parquet(ROOT/'inputs/case/evaluation_masks_v27.parquet'))
    ma=one_to_one(canonical,mm,POPS+['D_positive','O_positive'],'reporting_masks')
    masks={c:bool_array(ma[c]) for c in POPS+['D_positive','O_positive']}
    for p,(n,pos) in POP_COUNTS.items(): require((int(masks[p].sum()),int(y[masks[p]].sum()))==(n,pos),'POPULATION_COUNTS',p)
    require(np.array_equal(masks['D_positive']|masks['O_positive'],y==1),'POSITIVE_PARTITION_UNION')
    require(not (masks['D_positive']&masks['O_positive']).any(),'POSITIVE_PARTITION_OVERLAP')
    source=strict_keys(pd.read_parquet(ROOT/'inputs/financial/financial_status_masks_v27_frozen.parquet'))
    flagcols=read_json(package/'reference/PROTOCOL_v2.json')['columns']['flags3']
    sa=one_to_one(canonical,source,['target']+flagcols,'source_flags')
    require(np.array_equal(binary_array(sa.target),y),'FLAG_LABEL_IDENTITY')
    flags={c:bool_array(sa[c]).astype(float) for c in flagcols}
    inc=strict_keys(pd.read_parquet(ROOT/'inputs/case/positive_document_key_hash_incidence_v27.parquet'))
    require({'partition','doc_key_hash'}<=set(inc.columns),'INCIDENCE_SCHEMA')
    require(len(inc)==326 and not inc.duplicated(KEYS+['doc_key_hash']).any(),'INCIDENCE_COUNT_OR_DUPLICATE')
    require(inc.doc_key_hash.notna().all() and inc.doc_key_hash.astype(str).str.fullmatch('[0-9a-f]{64}').all(),'INCIDENCE_HASH_DOMAIN')
    all_lbl=inc.merge(labels[KEYS+['target','split_v1']],how='left',on=KEYS,validate='many_to_one',indicator=True)
    require((all_lbl['_merge']=='both').all() and all_lbl.target.eq(1).all(),'INCIDENCE_NONPOSITIVE_OR_MISSING_ROW')
    partition_map={'train_2010_2018':'train','validation_2019_2020':'validation','test_2021_2022':'test','right_censored_2023_2024':'outside'}
    require(np.array_equal(all_lbl.partition.to_numpy(str),all_lbl.split_v1.map(partition_map).to_numpy(str)),'INCIDENCE_PARTITION_CONFLICT')
    inc_test=inc[inc.partition=='test']
    require(len(inc_test[KEYS].drop_duplicates())==20,'INCIDENCE_TEST_POSITIVE_COVERAGE')

    lock=read_json(ROOT/'stage_t/ALL_105_LOCKED.json')
    require(lock['status']=='ALL_105_LOCKED' and lock['fit_count']==105,'GLOBAL_STAGE_T_LOCK')
    require(lock['final_binding_sha256']==EXPECTED['bindings/RUN_BINDINGS_v2.json'] and lock['run_matrix_sha256']==EXPECTED['bindings/sensitivity_run_matrix.json'],'GLOBAL_LOCK_PARENT_HASH')
    locks={r['run_id']:r for r in lock['entries']};require(len(locks)==105,'GLOBAL_LOCK_UNIQUE')
    manifest=read_json(ROOT/'stage_e_freeze_v1/SENSITIVITY_ARTIFACT_IDENTITIES.json')
    require(manifest==read_json(package/'reference/SENSITIVITY_ARTIFACT_IDENTITIES.json'),'MANIFEST_COPY_IDENTITY')
    recs={r['run_id']:r for r in manifest['records']}
    require(len(recs)==105 and set(recs)=={r['run_id'] for r in matrix},'E_MANIFEST_IDS')
    for parent,extra_allowed in [('stage_t',{'ALL_105_LOCKED.json'}),('stage_e',set())]:
        d=ROOT/parent
        require(d.is_dir(),'RUN_PARENT_MISSING',parent)
        dirs={p.name for p in d.iterdir() if p.is_dir()}
        require(dirs==set(recs),'RUN_DIRECTORY_SET',parent)
        require({p.name for p in d.iterdir() if not p.is_dir()}<=extra_allowed,'RUN_PARENT_EXTRA_FILES',parent)
    fits=[]
    for i,run in enumerate(matrix):
        rid=run['run_id'];r=recs[rid];t=ROOT/'stage_t'/rid;e=ROOT/'stage_e'/rid
        ledger.verify(t/'STAGE_T_COMPLETE.json',locks[rid]['complete_sha256'],role=rid+'/T_complete')
        ledger.verify(t/'FIT_IDENTITY.json',locks[rid]['fit_identity_sha256'],role=rid+'/T_identity')
        done=read_json(t/'STAGE_T_COMPLETE.json');ident=read_json(t/'FIT_IDENTITY.json')
        require(done.get('status')=='LOCKED' and done.get('run_id')==rid,'T_COMPLETION_CONTRACT')
        require(done.get('test_labels_read') is False and done.get('test_scores_computed') is False,'T_FLAGS')
        for name,meta in done['artifacts'].items():
            require(Path(name).name==name,'UNSAFE_ARTIFACT_NAME')
            ledger.verify(t/name,meta['sha256'],meta['size_bytes'],role=rid+'/T/'+name)
        require({'stage_t_result.json','FIT_IDENTITY.json','validation_predictions.parquet'}<=set(done['artifacts']),'T_MISSING_REQUIRED_ARTIFACTS')
        iid=ident['identity'];calc=hashlib.sha256((json.dumps(iid,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
        require(ident['identity_sha256']==calc and iid['run']==run,'T_FIT_IDENTITY')
        require(iid.get('final_binding_sha256')==EXPECTED['bindings/RUN_BINDINGS_v2.json'] and iid.get('runner_sha256')==EXPECTED['code/sensitivity_runner.py'],'T_BINDING_CHAIN')
        require(iid.get('stage')=='T' and iid.get('test_labels_read') is False and iid.get('test_scores_computed') is False,'T_STAGE_CONTRACT')
        tresult=read_json(t/'stage_t_result.json');thr=float(tresult['val_threshold'])
        require(np.isfinite(thr) and 0<=thr<=1,'THRESHOLD_DOMAIN')
        require(tresult.get('test_evaluated') is False,'T_TEST_FLAG')
        for key in ['run_id','family','model','modality','model_seed','imputation_seed']:
            require(tresult.get(key)==run.get(key),'T_RESULT_RUN_MISMATCH',key)
        ledger.verify(e/'STAGE_E_COMPLETE.json',r['stage_e_complete_sha256'],role=rid+'/E_complete')
        ledger.verify(e/'stage_e_result.json',r['stage_e_result_sha256'],role=rid+'/E_result')
        ledger.verify(e/'test_predictions.parquet',r['test_predictions_sha256'],r['test_predictions_size_bytes'],role=rid+'/E_prediction')
        edone=read_json(e/'STAGE_E_COMPLETE.json')
        require(edone.get('status')=='LOCKED' and edone.get('run_id')==rid,'E_COMPLETION_CONTRACT')
        require(set(edone.get('artifacts',{}))=={'stage_e_result.json','test_predictions.parquet'},'E_ARTIFACT_SET')
        for name,meta in edone['artifacts'].items():ledger.verify(e/name,meta['sha256'],meta['size_bytes'],role=rid+'/E/'+name)
        er=read_json(e/'stage_e_result.json')
        require(er.get('run_id')==rid and float(er['validation_threshold'])==thr,'E_THRESHOLD_OR_RUN_MISMATCH')
        scores=prediction(e/'test_predictions.parquet',canonical)
        fits.append(Fit(rid,condition_name(run),run['model_seed'],run['family'],run['model'],run['modality'],run['imputation_seed'],scores,thr,er))
        if (i+1)%20==0 or i==104:progress(f'Bound V27 frozen fits: {i+1}/105',flush=True)

    private=read_json(ROOT/'bindings/P2A_PRIVATE_REFERENCE_PATHS.json')
    original={}
    for r in private['bound']:
        key=(r['condition'],int(r['seed']))
        require(key not in original,'DUPLICATE_OLD_REFERENCE_KEY')
        original[key]=r
    spec=read_json(package/'reference/REFERENCE_EXPECTATIONS.json')
    require(len(spec)==len(original)==40,'OLD_REFERENCE_COUNT')
    for rec in spec:
        key=(rec['private_condition'],rec['seed']);require(key in original,'OLD_REFERENCE_MISSING')
        paths=original[key]
        rp=ledger.verify(Path(paths['result']),rec['result']['sha256'],rec['result']['bytes'],role=str(key)+'/old_result')
        pp=ledger.verify(Path(paths['predictions']),rec['prediction']['sha256'],rec['prediction']['bytes'],role=str(key)+'/old_prediction')
        if 'attempt' in rec:require(paths.get('attempt')==rec['attempt'],'OLD_REVERSE_ATTEMPT_IDENTITY')
        old=read_json(rp);thr=float(old['val_threshold'])
        require(old.get('model')==rec['model'] and old.get('modal')==rec['modality'] and int(old.get('seed',-1))==rec['seed'],'OLD_RESULT_MODEL_IDENTITY')
        require(old.get('test_evaluated') is True and np.isfinite(thr) and 0<=thr<=1,'OLD_EVALUATION_RECORD')
        scores=prediction(pp,canonical)
        fits.append(Fit('REF__'+rec['private_condition']+'__'+str(rec['seed']),rec['condition'],rec['seed'],'REFERENCE',rec['model'],rec['modality'],None,scores,thr,old))
    require(len(fits)==145,'TOTAL_FROZEN_FITS')
    for c in set(f.condition for f in fits):
        require(sorted(f.seed for f in fits if f.condition==c)==sorted(SEEDS),'CONDITION_SEED_SET',c)
    require(len({f.condition for f in fits})==29,'CONDITION_COUNT')
    return Bundle(canonical,y,masks,flags,inc,fits,final,data,ledger,matrix,EXPECTED['bindings/RUN_BINDINGS_v2.json'])
