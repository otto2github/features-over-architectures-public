"""Narrow adapters: frozen annual cache, common node universe, validation-only fits."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from e7_runtime_common import require


def strict_seed(seed):
    import numpy as np
    import torch
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8','CUBLAS_WORKSPACE_CONFIG_MISMATCH')
    random.seed(seed);np.random.seed(seed)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True,warn_only=False)
    require(torch.are_deterministic_algorithms_enabled()
            and not torch.is_deterministic_algorithms_warn_only_enabled(),'STRICT_DETERMINISM_MISMATCH')


def load_train_validation_df(data):
    """Same original join/order/conversions, physically reading only years 2010-2020."""
    import pyarrow.parquet as pq
    years = list(range(2010,2021))
    nf = pq.read_table(data/'node_features_v1_1.parquet',filters=[('year','in',years)]).to_pandas()
    labels = pq.read_table(data/'fraud_labels_v1_0.parquet',
        columns=['company_code','fiscal_year','split_v1','label_v1_strict_ab_primary'],
        filters=[('fiscal_year','in',years)]).to_pandas()
    nf['company_code'] = nf.firm_id.astype(str).str.zfill(6)
    labels = labels.rename(columns={'label_v1_strict_ab_primary':'target'})
    joined = nf.merge(labels,left_on=['company_code','year'],right_on=['company_code','fiscal_year'],
                      how='left',validate='one_to_one')
    require(len(joined)==32291 and joined.year.isin(years).all(),'TRAIN_VALIDATION_CONTEXT_ROWS')
    require(len(joined[joined.year<=2018])==24527 and len(joined[joined.year>=2019])==7764,
            'ANNUAL_TRAIN_VALIDATION_CONTEXT_COUNTS')
    return joined


def validate_supervision(core,df):
    for years,n,pos in [(core.TRAIN_YEARS,24106,254),(core.VAL_YEARS,7309,40)]:
        selected = df[df.year.isin(years) & df.target.notna()]
        require(len(selected)==n and selected.target.isin([0,1]).all()
                and int(selected.target.sum())==pos,'ORIGINAL_SUPERVISION_MISMATCH')
    require(core.dynamic_pos_weight(df)[1:]==(23852,254),'TRAIN_POS_WEIGHT_MISMATCH')


def check_scaler(core,df,binding,columns):
    import numpy as np
    require(core.select_cols(df,'M11')==columns,'M11_COLUMN_ORDER_MISMATCH')
    mu,sd = core.fit_scaler(df,columns)
    require(np.isfinite(mu).all() and np.isfinite(sd).all(),'NONFINITE_SCALER')
    require(hashlib.sha256(mu.tobytes()).hexdigest()==binding['scaler_mean_float32_sha256']
            and hashlib.sha256(sd.tobytes()).hexdigest()==binding['scaler_std_float32_sha256'],
            'FROZEN_SCALER_BYTES_MISMATCH')
    return mu,sd


def make_cache(original,condition,e7_edges,binding,years):
    import numpy as np
    import torch
    require(condition in {'control','E7'},'UNKNOWN_CONDITION')
    selected_years = list(years)
    def prepare(df,modal,data_dir):
        require(modal=='M11','ONLY_M11_ALLOWED')
        cache,n,d = original(df,modal,data_dir)
        require(n==binding['expected_n_original'] and d==129,'ORIGINAL_CACHE_SHAPE')
        selected = {}
        for year in selected_years:
            c = dict(cache[year])
            ei,et = c['edge_index'],c['edge_type']
            require(ei.shape[1]==binding['base_annual_edges'][str(year)],'BASE_YEAR_EDGE_COUNT')
            base = torch.cat((ei,ei.flip(0)),1)
            base_type = torch.cat((et,et),0)
            if condition=='E7':
                extra = e7_edges[e7_edges.year==year]
                require(len(extra)==binding['E7_annual_edges'][str(year)],'E7_YEAR_EDGE_COUNT')
                forward = torch.from_numpy(np.vstack([extra.src_idx,extra.dst_idx]).astype(np.int64))
                extra_type = torch.full((2*len(extra),),5,dtype=torch.long)
                c['edge_index'] = torch.cat((base,forward,forward.flip(0)),1)
                c['edge_type'] = torch.cat((base_type,extra_type),0)
            else:
                c['edge_index'],c['edge_type'] = base,base_type
            require(torch.equal(c['edge_index'][:,:base.shape[1]],base)
                    and torch.equal(c['edge_type'][:len(base_type)],base_type),'BASE_MESSAGE_PREFIX_CHANGED')
            require(c['company_idx'].numel()>0 and int(c['company_idx'].max())<n,'COMPANY_SUPPORT_INDEX_OVERLAP')
            require(c['company_x'].shape[1]==129,'COMPANY_FEATURE_DIMENSION')
            selected[year] = c
        return selected,binding['expected_n_extended'],d
    return prepare


class ValidationObserver:
    """Never permits test evaluation through the training-loop hook."""
    def __init__(self,original,auc_function,years,run_id):
        self.original,self.auc_function = original,auc_function
        self.years,self.run_id = list(years),run_id
        self.history=[];self.last=None;self.model=None

    def __call__(self,model,cache,years,n,d,device):
        import numpy as np
        require(list(years)==self.years,'TEST_EVALUATION_FORBIDDEN_DURING_STAGE_T')
        y,s,keys = self.original(model,cache,years,n,d,device)
        require(len(y)==7309 and int(np.sum(y))==40 and np.isfinite(s).all(),'VALIDATION_SCORE_COUNTS_OR_VALUES')
        auc=float(self.auc_function(y,s))
        self.history.append(auc);self.last=(y.copy(),s.copy(),keys.copy());self.model=model
        call=len(self.history)
        if call==1 or call%5==0:
            print(f'VALIDATION {self.run_id} call={call} auc={auc:.6f}',flush=True)
        return y,s,keys
