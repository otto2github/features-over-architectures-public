"""Annual distinct-company degrees; no graph rewiring or identity inference."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .common import *

ALIASES={'E1':'E1','E1_HOLDS_BY':'E1','E2':'E2','E2_FLOAT_HELD_BY':'E2','E3':'E3','E3_HAS_MANAGER':'E3','E4':'E4','E4_CO_HELD':'E4','E5':'E5','E5_CO_MGR':'E5'}

def graph_audit(ctx):
    labels=load_labels(ctx);nf=load_features(ctx)
    mp=ctx.find_pinned('node_id_index_label_v1_0_fiverel_derived.parquet')
    ep=ctx.find_pinned('global_edge_index.parquet')
    mapping=pd.read_parquet(mp,columns=['node_id','node_idx']).sort_values('node_idx').reset_index(drop=True)
    require(len(mapping)==ctx.p['expected_nodes'],'GRAPH_MAPPING_COUNT')
    require(np.array_equal(mapping.node_idx.to_numpy(),np.arange(len(mapping))),'GRAPH_MAPPING_NOT_DENSE')
    require(not mapping.node_id.duplicated().any(),'GRAPH_MAPPING_DUPLICATE_NODE')
    nodeid=mapping.node_id.astype(str)
    types=np.select([nodeid.str.startswith('C:'),nodeid.str.startswith('P:'),nodeid.str.startswith('I:')],['company','person','institution'],default='unknown')
    e=pd.read_parquet(ep,columns=['src_idx','dst_idx','edge_type','year'])
    require(len(e)==ctx.p['expected_edges'],'GRAPH_EDGE_COUNT')
    require(set(e.edge_type.astype(str))<=set(ALIASES),'GRAPH_UNKNOWN_RELATION')
    e=e.copy();e['relation']=e.edge_type.astype(str).map(ALIASES)
    for c in ['src_idx','dst_idx','year']:
        require(pd.to_numeric(e[c],errors='coerce').notna().all(),'GRAPH_NONNUMERIC_COLUMN:'+c)
        require(np.equal(e[c],e[c].astype(np.int64)).all(),'GRAPH_NONINTEGER_COLUMN:'+c)
        e[c]=e[c].astype(np.int64)
    require(e[['src_idx','dst_idx']].min().min()>=0 and e[['src_idx','dst_idx']].max().max()<len(mapping),'GRAPH_INDEX_RANGE')
    stats=[];hubs=[];inv=[]
    for year,g in e.groupby('year',sort=True):
        for rel,rr in g.groupby('relation'):
            inv.append({'fiscal_year':int(year),'relation':rel,'edge_records':len(rr),'consumed_in_current_protocol':2010<=year<=2022})
        bip=g[g.relation.isin(['E1','E2','E3'])]
        require(np.all(types[bip.src_idx.to_numpy()]=='company'),'BIPARTITE_SOURCE_NOT_COMPANY')
        require(np.isin(types[bip.dst_idx.to_numpy()],['person','institution']).all(),'BIPARTITE_DESTINATION_UNCLASSIFIED')
        unique=bip[['src_idx','dst_idx']].drop_duplicates()
        deg=unique.groupby('dst_idx').src_idx.nunique()
        e3=bip[bip.relation=='E3']
        e3person=e3[types[e3.dst_idx.to_numpy()]=='person']
        for typ in ['person','institution']:
            d=deg[types[deg.index.to_numpy()] == typ]
            stats.append({'fiscal_year':int(year),'node_type':typ,'graph_scope':'E1_E2_E3_union','degree_unit':'distinct_company_neighbors','active_entity_nodes':len(d),**quantiles(d.to_numpy())})
        # E3-only degree separates manager incidence from shareholder incidence.
        dm=e3person.groupby('dst_idx').src_idx.nunique()
        stats.append({'fiscal_year':int(year),'node_type':'person','graph_scope':'E3_only','degree_unit':'distinct_company_neighbors','active_entity_nodes':len(dm),**quantiles(dm.to_numpy())})
        for scope,d in [('E1_E2_E3_union',deg[types[deg.index.to_numpy()]=='person']),('E3_only',dm)]:
            for threshold in [20,50]:
                high=set(d[d>threshold].index)
                raw_hits=int(e3person.dst_idx.isin(high).sum())
                up=e3person[['src_idx','dst_idx']].drop_duplicates()
                uniq_hits=int(up.dst_idx.isin(high).sum())
                hubs.append({'fiscal_year':int(year),'degree_scope':scope,'strict_degree_threshold':threshold,'active_person_nodes':len(d),'person_nodes_above_threshold':len(high),'person_node_fraction_above_threshold':len(high)/len(d) if len(d) else None,'all_E3_records':len(e3),'person_E3_records':len(e3person),'person_E3_records_incident_on_high_degree_nodes':raw_hits,'fraction_of_all_E3_records':raw_hits/len(e3) if len(e3) else None,'fraction_of_person_E3_records':raw_hits/len(e3person) if len(e3person) else None,'unique_person_company_E3_pairs':len(up),'high_degree_unique_E3_pairs':uniq_hits,'messages_weighted_by_stored_multiplicity':True})
    ctx.export_csv('30_graph_annual_relation_counts.csv',inv)
    ctx.export_csv('31_graph_entity_degree_distributions.csv',stats)
    ctx.export_csv('32_graph_person_hub_E3_shares.csv',hubs)
    # Active-node coverage of the exact training-year undirected union, excluding self-loops.
    tr=e[e.year.between(2010,2018)&e.src_idx.ne(e.dst_idx)]
    active=np.unique(np.r_[tr.src_idx.to_numpy(),tr.dst_idx.to_numpy()])
    node_to_idx=dict(zip(mapping.node_id.astype(str),mapping.node_idx.astype(int)))
    f=nf[['company_code','fiscal_year','node_id']].merge(labels[['company_code','fiscal_year','target','partition']],on=['company_code','fiscal_year'],validate='one_to_one')
    idx=f.node_id.astype(str).map(node_to_idx)
    require(idx.notna().all(),'COVERAGE_COMPANY_NODE_NOT_MAPPED')
    f['covered']=np.isin(idx.to_numpy(dtype=int),active)
    rows=[]
    for part,g in f[f.target.notna()].groupby('partition'):
        rows.append({'partition':part,'eligible_rows':len(g),'covered_rows':int(g.covered.sum()),'uncovered_rows':int((~g.covered).sum()),'coverage_fraction':float(g.covered.mean()),'positive_rows':int((g.target==1).sum()),'covered_positive_rows':int(g.loc[g.target==1,'covered'].sum()),'source':'recomputed from frozen graph and mapping; not transcribed terminal output'})
    ctx.export_csv('33_training_union_coverage_RECOMPUTED.csv',rows)
    ctx.export_json('34_graph_interpretation_limits.json',{'node_type_source':'C:/P:/I: prefix in pinned mapping from retained graph builder','degree_does_not_prove_false_name_merge':True,'distinct_homonyms_independently_identified':False,'names_or_node_identifiers_exported':False,'edges_deleted_or_modified':False,'self_loops_excluded_for_training_union_coverage':True,'annual_hub_stats_use_original_E1_E2_E3_records_not_appended_reverse_duplicates':True})
    ctx.record('graph_hubs_and_coverage','PASS',note='Distinct-company degree and record-weighted E3 shares; no claim that high degree proves mistaken identity.')
