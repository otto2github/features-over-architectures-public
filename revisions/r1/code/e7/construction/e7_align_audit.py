#!/usr/bin/env python3
"""4090: frozen company/year alignment and no-fit graph operator audit."""
from __future__ import annotations
from collections import Counter
import gzip
import hashlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
from pathlib import Path
import random
import sys
import time
import zipfile

sys.dont_write_bytecode = True
from e7_prep_common import KIT, YEARS, require, sha, verify_records, write_json, zip_exact

ROOT = Path('/path/to/private_workspace/baseline')
RETURN_FILES = ['PREP_SUMMARY.json','ANNUAL_ALIGNMENT.csv','OPERATOR_CHECKS.json',
                'ENVIRONMENT.json','FROZEN_INPUT_IDENTITIES.json','M11_COLUMN_ORDER.json',
                'BUILD_SUMMARY.json','RUNNER_IDENTITIES.json','README_RETURN.txt']


def read_bridge(path, reference):
    import pandas as pd
    with zipfile.ZipFile(path) as book:
        require(book.namelist() == ['PRIVATE_BRIDGE_MANIFEST.json','e7_record_incidence.csv.gz'],
                'PRIVATE_BRIDGE_MEMBER_LIST_MISMATCH')
        require(all(x.file_size <= 40*1024**2 for x in book.infolist()), 'BRIDGE_SIZE_LIMIT')
        require(book.testzip() is None, 'PRIVATE_BRIDGE_ZIP_INVALID')
        manifest = json.loads(book.read('PRIVATE_BRIDGE_MANIFEST.json'))
        require(manifest['status'] == 'PRIVATE_RECORD_GRAPH_BUILT_ALIGNMENT_PENDING'
                and manifest['graph_rule_id'] == reference['graph_rule_id']
                and manifest['protocol_sha256'] == sha(KIT / 'E7_GRAPH_PREP_PROTOCOL_v1.md')
                and manifest['source_identities'] == reference['tencent_sources'],
                'PRIVATE_BRIDGE_PROTOCOL_OR_SOURCE_MISMATCH')
        compressed = book.read('e7_record_incidence.csv.gz')
    require(hashlib.sha256(compressed).hexdigest() == manifest['incidence_file']['sha256']
            and len(compressed) == manifest['incidence_file']['size_bytes'], 'INCIDENCE_HASH_MISMATCH')
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        raw = stream.read(40*1024**2+1)
    require(len(raw) <= 40*1024**2, 'DECOMPRESSED_INCIDENCE_SIZE_LIMIT')
    edges = pd.read_csv(io.BytesIO(raw), dtype={'firm_code':str,'party_key':str,'year':'int64'})
    require(list(edges.columns) == ['year','firm_code','party_key'], 'INCIDENCE_SCHEMA_MISMATCH')
    require(len(edges) == reference['selected_primary_edge_total'] and edges.notna().all().all()
            and edges.year.isin(YEARS).all()
            and edges.firm_code.str.fullmatch(r'[0-9]{6}').all()
            and edges.party_key.str.fullmatch(r'[0-9a-f]{64}').all()
            and not edges.duplicated().any(), 'INCIDENCE_VALUES_INVALID')
    require(manifest['annual_profiles_before_frozen_feature_panel_alignment']
            == reference['selected_annual_profile_oracle'], 'BRIDGE_PROFILE_MISMATCH')
    return edges, manifest


def align_company_year(incidence, features, mapping):
    """Only feature-bearing issuer-years; no guessed C:code/exchange mapping."""
    import numpy as np
    import pandas as pd
    require({'firm_id','node_id','year'} <= set(features), 'FEATURE_ID_COLUMNS_MISSING')
    require({'node_id','node_idx'} <= set(mapping), 'MAPPING_COLUMNS_MISSING')
    require(mapping.node_id.notna().all() and not mapping.node_id.astype(str).duplicated().any(),
            'MAPPING_NODE_ID_NOT_UNIQUE')
    idx = mapping.node_idx.to_numpy()
    require(np.isfinite(idx).all() and (idx == idx.astype(np.int64)).all(), 'MAPPING_INDEX_NOT_INTEGER')
    require(np.array_equal(np.sort(idx.astype(np.int64)), np.arange(len(mapping))),
            'MAPPING_INDEX_NOT_CONTIGUOUS')
    company = features[['firm_id','node_id','year']].copy()
    # Exact frozen core conversion, not an alternative code-normalization routine.
    company['firm_code'] = company.firm_id.astype(str).str.zfill(6)
    require(company.firm_code.str.fullmatch(r'[0-9]{6}').all() and company.node_id.notna().all(),
            'FROZEN_CORE_COMPANY_ID_CONVERSION_INVALID')
    require(not company.duplicated(['firm_code','year']).any(), 'DUPLICATE_FROZEN_COMPANY_YEAR')
    company['node_id'] = company.node_id.astype(str)
    node_to_idx = dict(zip(mapping.node_id.astype(str), mapping.node_idx.astype(int)))
    company['src_idx'] = company.node_id.map(node_to_idx)
    require(company.src_idx.notna().all(), 'FEATURE_COMPANY_NODE_ABSENT_FROM_FROZEN_MAPPING')
    require(not company.duplicated(['year','src_idx']).any(), 'MULTIPLE_COMPANIES_SHARE_ONE_ANNUAL_NODE')
    aligned = incidence.merge(company[['firm_code','year','src_idx']],
                              on=['firm_code','year'], how='left', validate='many_to_one')
    dropped = aligned[aligned.src_idx.isna()].copy()
    kept = aligned[aligned.src_idx.notna()].copy()
    require(len(kept) > 0, 'NO_RPT_EDGES_MATCH_FROZEN_COMPANY_YEARS')
    kept['src_idx'] = kept.src_idx.astype(np.int64)
    keys = sorted(kept.party_key.unique())
    support = pd.DataFrame({'node_id':['E7R:'+x for x in keys],
                            'node_idx':np.arange(len(mapping), len(mapping)+len(keys), dtype=np.int64)})
    require(not set(support.node_id) & set(mapping.node_id.astype(str)), 'SUPPORT_NAMESPACE_COLLISION')
    key_to_idx = dict(zip(keys, support.node_idx))
    kept['dst_idx'] = kept.party_key.map(key_to_idx).astype(np.int64)
    kept['edge_type'] = 'E7_RPT_RECORD'
    kept = kept.sort_values(['year','src_idx','dst_idx'], kind='stable').reset_index(drop=True)
    require(not kept.duplicated(['year','src_idx','dst_idx']).any(), 'DUPLICATE_ALIGNED_E7_EDGE')
    require((kept.src_idx < len(mapping)).all() and (kept.dst_idx >= len(mapping)).all(),
            'SUPPORT_INDEX_OVERLAP')
    stats = []
    for year in YEARS:
        before, after, missing = [x[x.year == year] for x in (incidence, kept, dropped)]
        degrees = after.groupby('party_key').firm_code.nunique()
        shared = set(degrees[degrees >= 2].index)
        cohort = company[company.year == year]
        parent = {int(x):int(x) for x in after.src_idx.unique()}
        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node
        previous,first = None,None
        for party,node in after[['party_key','src_idx']].sort_values(['party_key','src_idx']).itertuples(index=False,name=None):
            node = int(node)
            if party != previous:
                previous,first = party,node
            else:
                a,b = find(first),find(node)
                if a != b:
                    parent[b] = a
        components = Counter(find(node) for node in parent)
        largest = max(components.values(),default=0)
        stats.append({'year':year, 'candidate_edges_before_alignment':len(before),
          'aligned_forward_edges':len(after), 'dropped_absent_feature_year_edges':len(missing),
          'dropped_absent_feature_year_issuers':missing.firm_code.nunique(),
          'feature_panel_companies':len(cohort), 'feature_panel_companies_with_e7':after.firm_code.nunique(),
          'retained_party_keys':len(degrees), 'degree_one_party_keys':int((degrees == 1).sum()),
          'shared_party_keys':int((degrees >= 2).sum()),
          'companies_with_shared_party':after[after.party_key.isin(shared)].firm_code.nunique(),
          'maximum_party_issuer_degree':int(degrees.max()) if len(degrees) else 0,
          'rpt_only_issuer_component_count':len(components),
          'rpt_only_largest_issuer_component':largest})
    return kept, support, stats


def strict_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    require(torch.are_deterministic_algorithms_enabled()
            and not torch.is_deterministic_algorithms_warn_only_enabled(), 'STRICT_DETERMINISM_NOT_ENABLED')


def append_reverse(ei):
    import torch
    return torch.cat((ei, ei.flip(0)), dim=1)


def operator_checks(core, joined, aligned, n_extended):
    import numpy as np
    import torch
    cache, n_original, dim = core.prepare_graph_cache(joined, 'M11', ROOT / 'data')
    require(n_original == 303775 and dim == 129, 'BASELINE_CACHE_NODE_OR_FEATURE_DIMENSION')
    checks, sources = [], {}
    for name in ['GCN','SAGE']:
        strict_seed(42)
        model = core.make_model(name, dim, 64, 5).to('cuda').eval()
        for layer in [model.c1,model.c2]:
            require(layer.flow == 'source_to_target', 'MODEL_MESSAGE_FLOW_MISMATCH')
            source = Path(inspect.getsourcefile(type(layer)))
            sources[type(layer).__name__] = {'source_basename':source.name,'sha256':sha(source)}
        for year in YEARS:
            c = cache[year]
            ei = append_reverse(c['edge_index']).to('cuda')
            x = torch.zeros((n_original,dim), dtype=torch.float32, device='cuda')
            x[c['company_idx'].to('cuda')] = c['company_x'].to('cuda')
            require(torch.isfinite(x).all().item(), 'NONFINITE_FROZEN_INPUT')
            with torch.no_grad():
                old = model(x,ei)
                extended_x = torch.cat((x,torch.zeros((n_extended-n_original,dim),device='cuda')),0)
                extended = model(extended_x,ei)
            diff = float((old-extended[:n_original]).abs().max().item())
            require(torch.isfinite(old).all().item() and torch.isfinite(extended).all().item()
                    and torch.allclose(old,extended[:n_original],rtol=1e-6,atol=1e-6),
                    f'ISOLATED_EXTENSION_OPERATOR_PARITY_FAILED:{name}:{year}')
            checks.append({'model':name,'year':year,'check':'isolated_extension_eval_parity',
                           'max_abs_original_node_logit_difference':diff,'atol':1e-6,'rtol':1e-6,'pass':True})
            print(f'Operator parity {name} {year}: PASS (max difference {diff:.3g})', flush=True)
            del x,extended_x,old,extended,ei
        # Technical full-graph backward probe: no labels, loss, optimizer or parameter updates.
        c, year = cache[2018], 2018
        base_ei = append_reverse(c['edge_index']).to('cuda')
        extra = aligned[aligned.year == year]
        extra_ei = append_reverse(torch.from_numpy(np.vstack([extra.src_idx,extra.dst_idx]).astype(np.int64))).to('cuda')
        for condition, ei in [('control',base_ei),('E7',torch.cat((base_ei,extra_ei),1))]:
            digests = []
            for repeat in [1,2]:
                strict_seed(42)
                probe = core.make_model(name,dim,64,5).to('cuda').train()
                x = torch.zeros((n_extended,dim),dtype=torch.float32,device='cuda')
                x[c['company_idx'].to('cuda')] = c['company_x'].to('cuda')
                logits = probe(x,ei)
                require(torch.isfinite(logits).all().item(), 'BACKWARD_PROBE_NONFINITE_OUTPUT')
                logits[c['company_idx'].to('cuda')].square().mean().backward()
                h = hashlib.sha256()
                for key,p in probe.named_parameters():
                    require(p.grad is not None and torch.isfinite(p.grad).all().item(), 'INVALID_PROBE_GRADIENT')
                    h.update(key.encode());h.update(p.grad.detach().cpu().numpy().tobytes())
                digests.append(h.hexdigest())
                del x,probe,logits
            require(digests[0] == digests[1], 'STRICT_BACKWARD_REPEAT_MISMATCH:' + name + ':' + condition)
            checks.append({'model':name,'year':year,'condition':condition,
                           'check':'strict_full_graph_forward_backward_repeat','gradient_sha256':digests[0],
                           'optimizer_steps':0,'labels_used':False,'pass':True})
            print(f'Strict backward {name} {condition}: PASS', flush=True)
        del model,base_ei,extra_ei
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return {'status':'PASS','checks':checks,'operator_source_identities':sources,
            'formal_models_fitted':0,'parameters_optimized':False,
            'test_labels_used_for_operator_checks':False,'test_performance_evaluated':False}


def run(output):
    started = time.monotonic()
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import torch
    import torch_geometric.backend
    import torch_geometric.typing
    reference = json.loads((KIT / 'INPUT_BINDINGS.json').read_text())
    records = reference['gpu_baseline_sources']
    verify_records(records)
    bridge = KIT / 'E7_PRIVATE_BRIDGE.zip'
    require(bridge.is_file() and not bridge.is_symlink(), 'PRIVATE_BRIDGE_MISSING')
    bridge_sha = sha(bridge)
    incidence, build = read_bridge(bridge, reference)
    versions = {}
    for name in [*reference['baseline_environment_versions'],'pyg-lib','torch-scatter','torch-sparse']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    optional = {key:bool(getattr(torch_geometric.typing,key,False)) for key in
                ['WITH_PYG_LIB','WITH_SEGMM','WITH_TORCH_SCATTER','WITH_TORCH_SPARSE']}
    before = torch_geometric.backend.use_segment_matmul
    torch_geometric.backend.use_segment_matmul = False
    env = {'python':sys.version,'executable':sys.executable,'versions':versions,
           'cuda_available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda,
           'optional_backend_flags':optional,'segment_matmul_before':before,'segment_matmul_effective':False,
           'cudnn_version':torch.backends.cudnn.version(),'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
           'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}
    if env['cuda_available']:
        env.update({'gpu_name':torch.cuda.get_device_name(0),'gpu_total_memory':torch.cuda.get_device_properties(0).total_memory})
    blockers = []
    if sys.version_info[:2] != (3,12):
        blockers.append('Python major/minor differs from primary reference')
    for name,expected in reference['baseline_environment_versions'].items():
        if versions[name] != expected:
            blockers.append('Baseline package version mismatch: ' + name)
    if not env['cuda_available']:
        blockers.append('CUDA unavailable; no CPU fallback')
    if optional['WITH_TORCH_SCATTER'] or optional['WITH_TORCH_SPARSE']:
        blockers.append('Optional scatter/sparse backend differs from primary reference')
    spec = importlib.util.spec_from_file_location('e7_frozen_core',KIT / 'formal_rerun_core_REFERENCE.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    data = ROOT / 'data'
    features = pd.read_parquet(data / 'node_features_v1_1.parquet')
    mapping = pd.read_parquet(data / 'node_id_index_label_v1_0_fiverel_derived.parquet')
    require(len(features) == 51675 and len(mapping) == 303775, 'FROZEN_PANEL_OR_MAPPING_SIZE')
    joined = core.load_joined(data, 'label_v1_strict_ab_primary')
    cols = core.select_cols(joined,'M11')
    inventory = pd.read_csv(KIT / 'feature_columns_129_v5.csv')
    require(cols == inventory.sort_values('M11_order_1_based').column_name.tolist(), 'M11_COLUMN_ORDER_MISMATCH')
    mu,sd = core.fit_scaler(joined,cols)
    require(np.isfinite(mu).all() and np.isfinite(sd).all(), 'SCALER_NONFINITE')
    supervision = {}
    for split,years in [('train',core.TRAIN_YEARS),('validation',core.VAL_YEARS),('test',core.TEST_YEARS)]:
        eligible = joined[joined.year.isin(years) & joined.target.notna()]
        require(eligible.target.isin([0,1]).all(), 'SUPERVISION_NONBINARY')
        supervision[split] = {'n':len(eligible),'positive':int(eligible.target.sum())}
    require(supervision == {'train':{'n':24106,'positive':254},'validation':{'n':7309,'positive':40},
                            'test':{'n':8435,'positive':20}}, 'ORIGINAL_SUPERVISION_COUNTS_MISMATCH')
    edges = pd.read_parquet(data / 'global_edge_index.parquet',columns=['src_idx','dst_idx','edge_type','year'])
    require(len(edges) == 4469735 and not set(edges.edge_type.astype(str))-set(core.REL_MAP), 'BASE_RELATION_OR_COUNT_MISMATCH')
    for key in ['src_idx','dst_idx','year']:
        v = edges[key].to_numpy()
        require(np.isfinite(v).all() and (v == v.astype(np.int64)).all(), 'BASE_INTEGER_COLUMN_INVALID')
    require((edges[['src_idx','dst_idx']].to_numpy() >= 0).all()
            and (edges[['src_idx','dst_idx']].to_numpy() < len(mapping)).all(), 'BASE_EDGE_ENDPOINT_RANGE')
    require(not (edges.src_idx == edges.dst_idx).any(), 'UNEXPECTED_STORED_BASE_SELF_LOOP')
    require({str(y):int((edges.year == y).sum()) for y in YEARS} == reference['baseline_annual_edges'],
            'BASE_ANNUAL_EDGE_COUNTS_MISMATCH')
    aligned,support,stats = align_company_year(incidence,features,mapping)
    for row in stats:
        year = row['year']
        cohort = joined[(joined.year == year) & joined.target.notna()]
        exposed = set(aligned[aligned.year == year].firm_code)
        row.update({'eligible_companies':len(cohort),
                    'eligible_companies_with_e7':int(cohort.company_code.isin(exposed).sum()),
                    'base_forward_edges':reference['baseline_annual_edges'][str(year)],
                    'control_message_edges':2*reference['baseline_annual_edges'][str(year)],
                    'treatment_message_edges':2*(reference['baseline_annual_edges'][str(year)]+row['aligned_forward_edges'])})
    # These remain private. The aggregate return ZIP cannot contain them.
    aligned[['src_idx','dst_idx','edge_type','year']].to_parquet(output / 'PRIVATE_e7_forward_edges.parquet',index=False)
    support.to_parquet(output / 'PRIVATE_e7_support_index.parquet',index=False)
    schemas = {p.name:[{'name':f.name,'type':str(f.type)} for f in pq.ParquetFile(p).schema_arrow]
               for p in [data / x for x in ['node_features_v1_1.parquet','fraud_labels_v1_0.parquet',
                         'global_edge_index.parquet','node_id_index_label_v1_0_fiverel_derived.parquet']]}
    op = {'status':'NOT_RUN','reason':'; '.join(blockers),'formal_models_fitted':0}
    if not blockers:
        try:
            op = operator_checks(core,joined,aligned,len(mapping)+len(support))
        except Exception as exc:
            blockers.append('No-fit operator check failed: ' + type(exc).__name__ + ': ' + str(exc))
            op = {'status':'FAILED','reason':blockers[-1],'formal_models_fitted':0,
                  'test_performance_evaluated':False}
    verify_records(records)
    require(sha(bridge) == bridge_sha, 'PRIVATE_BRIDGE_CHANGED_DURING_AUDIT')
    private_ledger = {name:{'size_bytes':(output/name).stat().st_size,'sha256':sha(output/name)} for name in
                      ['PRIVATE_e7_forward_edges.parquet','PRIVATE_e7_support_index.parquet']}
    summary = {'status':'GRAPH_PREP_BLOCKED_REVIEW_REQUIRED' if blockers else 'GRAPH_PREP_COMPLETE_TRAINING_PROTOCOL_PENDING',
      'training_started':False,'training_ready':False,'new_model_fits':0,'test_performance_evaluated':False,
      'graph_rule_id':reference['graph_rule_id'],'protocol_sha256':sha(KIT / 'E7_GRAPH_PREP_PROTOCOL_v1.md'),
      'blockers':blockers,'global_legal_party_id_semantics_verified':False,
      'semantic_scope':'conservative ID/name record-incidence; not a validated legal-entity graph',
      'baseline_hashes_rechecked':True,'original_core_unchanged':True,
      'M11_dimensions':129,'support_feature_policy':'zero for all E7 support nodes in both conditions',
      'original_node_count':len(mapping),'E7_support_node_count':len(support),
      'common_extended_node_count':len(mapping)+len(support),'supervision':supervision,
      'candidate_edges_before_alignment':len(incidence),'aligned_forward_edges':len(aligned),
      'dropped_absent_frozen_feature_year_edges':len(incidence)-len(aligned),
      'scaler_mean_float32_sha256':hashlib.sha256(mu.tobytes()).hexdigest(),
      'scaler_std_float32_sha256':hashlib.sha256(sd.tobytes()).hexdigest(),
      'private_bridge_sha256':bridge_sha,'private_derived_artifacts':private_ledger,
      'private_derived_artifacts_in_return':False,'baseline_schema':schemas,
      'preserve_base_edge_order_and_multiplicity':True,
      'message_order':'base forward; base reverse; E7 forward; E7 reverse',
      'annual_alignment':stats,'elapsed_seconds':time.monotonic()-started}
    write_json(output / 'PREP_SUMMARY.json',summary)
    pd.DataFrame(stats).to_csv(output / 'ANNUAL_ALIGNMENT.csv',index=False)
    write_json(output / 'OPERATOR_CHECKS.json',op)
    write_json(output / 'ENVIRONMENT.json',env)
    write_json(output / 'FROZEN_INPUT_IDENTITIES.json',records)
    write_json(output / 'M11_COLUMN_ORDER.json',{'dimensions':129,'columns':cols})
    write_json(output / 'BUILD_SUMMARY.json',build)
    runner_names = set(json.loads((KIT/'PACKAGE_SHA256.json').read_text())['files']) | {'PACKAGE_SHA256.json'}
    write_json(output / 'RUNNER_IDENTITIES.json',{p.name:{'sha256':sha(p),'size_bytes':p.stat().st_size}
      for p in KIT.iterdir() if p.is_file() and p.name in runner_names})
    (output / 'README_RETURN.txt').write_text(
      'Private author audit, not a public release. Only aggregates, schemas and file identities.\n'
      'No model was fitted. All training/checkpoint/evaluation code still requires final protocol binding.\n'
      'The provider ID is not asserted to be a verified globally unique legal entity identifier.\n'
      'Licensed row-level edges and support indices remain on the research server, excluded from this ZIP.\n',encoding='utf-8')
    zip_exact(output / 'E7_PREP_RETURN.zip',output,RETURN_FILES)
    print(summary['status'] + '; NO_MODEL_FITTED',flush=True)
    return summary
