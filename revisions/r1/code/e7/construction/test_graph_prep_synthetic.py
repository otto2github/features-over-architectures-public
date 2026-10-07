"""Independent edge, alignment, source-failure and privacy fixtures; no CUDA here."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import zipfile
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
KIT = ROOT if (ROOT/'e7_build_private.py').is_file() else ROOT/'PeerJ_141707_V36_E7_GRAPH_PREP_20261006'
sys.path.insert(0,str(KIT))
import e7_build_private as build
import e7_align_audit as audit
import e7_prep_common as common


def workbook(path, rows, alias=False):
    headings = ['Stkcd','Reptdt','Repart','RelatedPartyID' if alias else 'RalatedPartyID']
    all_rows = [headings,['代码','日期','名称','标识'],['字符','日期','字符','字符']] + rows
    fragments = []
    for n,values in enumerate(all_rows,1):
        cells = []
        for i,value in enumerate(values):
            address = chr(65+i)+str(n)
            if isinstance(value,tuple):
                literal,kind = value
                cell = f'<c r="{address}" t="{kind}"><v>{escape(str(literal))}</v></c>'
            else:
                cell = f'<c r="{address}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'
            cells.append(cell)
        fragments.append(f'<row r="{n}">'+''.join(cells)+'</row>')
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('xl/workbook.xml','<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="one" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr('xl/worksheets/sheet1.xml','<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'+''.join(fragments)+'</sheetData></worksheet>')


def expected_profile(year, n_edges, n_firms, degrees, shared_firms, components, largest):
    return {'year':year,'unique_issuer_party_year_edges':n_edges,'unique_issuers':n_firms,
      'unique_party_keys':len(degrees),'party_keys_degree_1':degrees.count(1),
      'party_keys_degree_ge_2':sum(x>=2 for x in degrees),'party_keys_degree_ge_5':0,
      'party_keys_degree_ge_10':0,'party_keys_degree_ge_50':0,
      'max_party_distinct_issuer_degree':max(degrees),'issuers_with_shared_party':shared_firms,
      'issuer_shared_party_fraction':shared_firms/n_firms,
      'incidence_edges_to_shared_parties':sum(x for x in degrees if x>=2),
      'rpt_only_issuer_component_count':components,'rpt_only_largest_issuer_component':largest,
      'largest_component_fraction':largest/n_firms,'top_20_party_degrees_no_identifiers':degrees,
      'primary_graph_year_2010_2022':True}


def expect_failure(call, marker):
    try:
        call()
    except RuntimeError as exc:
        assert marker in str(exc), str(exc)
    else:
        raise AssertionError('Expected failure: '+marker)


checks = []
with tempfile.TemporaryDirectory(prefix='synthetic_',dir=ROOT) as tmp:
    root = Path(tmp);raw = root/'data/raw/csmar';raw.mkdir(parents=True)
    records = [
      [['1','2010-12-31','PrivacyAlphaName','P1000'],['2','2010-12-31','PrivacyAlphaName','P1000'],
       ['3','2010-12-31','PrivacyBetaName','P2000'],['4','2010-12-31','PrivacyGammaName','P2000'],
       ['5','2010-12-31','PrivacySelfAName','P3000'],['5','2010-12-31','PrivacySelfBName','P3000']],
      [['1','2010-12-31','PrivacyAlphaName','P1000'],['6','2010-12-31','关联方','P4000'],
       ['7','2010-12-31','PrivacyMissingName',''],['8','2010-12-31','PrivacyAlphaName','P5000'],
       ['9','2010-12-31','PrivacyAlphaName','001'],['10','2010-12-31','PrivacyAlphaName','1'],
       ['11','2010-12-31','PrivacyUnsafeName',('1234567890123456','n')],
       ['12','2010-12-31','PrivacyReservedName','unknown']],
      [['1','2011-12-31','PrivacyRenamedName','P1000'],['2','2011-12-31','PrivacyRenamedName','P1000'],
       ['3','2011-12-31','PrivacyBetaName','P2000'],['99','2023-12-31','PrivacyFutureName','P1000'],
       ['2','bad-date','PrivacyBadDateName','P6000']]]
    for n,(filename,rows) in enumerate(zip(build.reader.VOLUMES,records)):
        workbook(raw/filename,rows,alias=(n==1))
    before = {p.name:common.sha(p) for p in raw.iterdir()}
    oracle = [expected_profile(2010,5,5,[2,1,1,1],2,4,2),
              expected_profile(2011,3,3,[2,1],2,2,2)]
    total = Counter()
    with sqlite3.connect(root/'test.sqlite') as db:
        build.init_db(db)
        for n,filename in enumerate(build.reader.VOLUMES):
            total.update(build.read_volume(raw/filename,db,n))
        actual = build.select_edges(db)
        assert actual == oracle
        assert total['metadata_rows_skipped']==6 and total['data_rows']==19
        assert total['valid_issuer_date_rows']==18 and total['candidate_id_rows']==17
        assert total['primary_candidate_id_rows']==16
        assert db.execute('SELECT COUNT(*) FROM conflicts').fetchone()[0]==2
        build.write_incidence(root/'one.csv.gz',db)
        build.write_incidence(root/'two.csv.gz',db)
        assert (root/'one.csv.gz').read_bytes()==(root/'two.csv.gz').read_bytes()
    checks.append('Independent annual oracle: cross-issuer and within-issuer same-year conflicts excluded; later-year rename retained; 2023 alias cannot alter earlier-year keys; repeats deduplicated; leading zeros preserved; generic, missing, reserved and unsafe values excluded; alias header accepted')
    incidence = pd.read_csv(root/'one.csv.gz',dtype={'firm_code':str,'party_key':str})
    assert len(incidence)==8
    assert set(incidence[incidence.year==2010].party_key) & set(incidence[incidence.year==2011].party_key)==set()
    checks.append('Cross-year compound keys split changed names; gzip serialization deterministic')
    fake_kit = root/'kit';fake_kit.mkdir()
    (fake_kit/'E7_GRAPH_PREP_PROTOCOL_v1.md').write_text('synthetic protocol\n')
    bindings = {'graph_rule_id':'SYNTHETIC','tencent_sources':[{'path':str(p),'size_bytes':p.stat().st_size,'sha256':common.sha(p)} for p in sorted(raw.iterdir())],
       'focus_total_count_checks':{k:total[k] for k in ['metadata_rows_skipped','data_rows','valid_issuer_date_rows','candidate_id_rows','missing_or_sentinel_id_rows','candidate_id_with_clean_name_rows']},
       'selected_annual_profile_oracle':oracle,'primary_candidate_id_rows':16,'selected_primary_edge_total':8,
       'identity_return_sha256':'synthetic'}
    common.write_json(fake_kit/'INPUT_BINDINGS.json',bindings)
    saved_kit,saved_project = build.KIT,build.PROJECT
    build.KIT,build.PROJECT = fake_kit,root
    out = root/'complete';out.mkdir()
    build.run(out)
    assert not list(out.glob('private_work_*'))
    assert {p.name:common.sha(p) for p in raw.iterdir()}==before
    with zipfile.ZipFile(out/'E7_PRIVATE_BRIDGE.zip') as z:
        assert z.namelist()==['PRIVATE_BRIDGE_MANIFEST.json','e7_record_incidence.csv.gz']
        content = z.read('PRIVATE_BRIDGE_MANIFEST.json')+gzip.decompress(z.read('e7_record_incidence.csv.gz'))
        for marker in ['PrivacyAlphaName','PrivacyFutureName','P1000','P2000','P3000','1234567890123456']:
            assert marker.casefold().encode() not in content.lower()
    saved_audit_kit = audit.KIT;audit.KIT=fake_kit
    restored,manifest = audit.read_bridge(out/'E7_PRIVATE_BRIDGE.zip',bindings)
    assert restored.equals(incidence)
    checks.append('End-to-end synthetic private bridge binds sources and annual oracle; excludes literal party IDs/names; raw bytes unchanged; temp database removed')
    features = pd.DataFrame({'firm_id':['1','1','2','3','9'], 'year':[2010,2011,2010,2011,2010],
       'node_id':['C:OTHER_A.SH','C:OTHER_A.SH','C:OTHER_B.SZ','C:OTHER_C.BJ','C:OTHER_D.SZ']})
    mapping = pd.DataFrame({'node_id':['C:OTHER_A.SH','C:OTHER_B.SZ','old_support','C:OTHER_C.BJ','C:OTHER_D.SZ'],
                            'node_idx':[3,0,4,2,1]})
    kept,support,annual = audit.align_company_year(incidence,features,mapping)
    assert len(kept)==5 and len(support)==4
    assert set(kept[kept.firm_code=='000001'].src_idx)=={3}
    assert set(kept[(kept.firm_code=='000002') & (kept.year==2010)].src_idx)=={0}
    assert not ((kept.firm_code=='000002') & (kept.year==2011)).any()
    assert list(support.node_idx)==[5,6,7,8]
    assert list(support.node_id)==sorted(support.node_id)
    assert annual[0]['dropped_absent_feature_year_edges']==2 and annual[1]['dropped_absent_feature_year_edges']==1
    assert annual[0]['rpt_only_largest_issuer_component']==2 and annual[1]['rpt_only_largest_issuer_component']==1
    assert set(kept.src_idx).isdisjoint(set(kept.dst_idx))
    checks.append('Frozen annual feature-to-node mapping handles arbitrary node IDs and row order; known company absent that year excluded; missing issuer-years counted; support indices sorted, contiguous and disjoint')
    expect_failure(lambda:audit.align_company_year(incidence,pd.concat([features,features.iloc[[0]]]),mapping),'DUPLICATE_FROZEN_COMPANY_YEAR')
    badmapping=mapping.copy();badmapping.loc[0,'node_idx']=9
    expect_failure(lambda:audit.align_company_year(incidence,features,badmapping),'MAPPING_INDEX_NOT_CONTIGUOUS')
    badmapping=mapping.copy();badmapping.loc[0,'node_id']='missing'
    expect_failure(lambda:audit.align_company_year(incidence,features,badmapping),'FEATURE_COMPANY_NODE_ABSENT')
    badmapping=mapping.copy();badmapping.loc[2,'node_id']=support.node_id.iloc[0]
    expect_failure(lambda:audit.align_company_year(incidence,features,badmapping),'SUPPORT_NAMESPACE_COLLISION')
    checks.append('Fail closed on duplicate company-years, invalid node indices, missing frozen company nodes and support namespace collision')
    badbridge=root/'badbridge.zip'
    with zipfile.ZipFile(out/'E7_PRIVATE_BRIDGE.zip') as source,zipfile.ZipFile(badbridge,'w') as dest:
        for name in source.namelist():
            data=source.read(name)
            if name.endswith('.gz'):data+=b'corrupted'
            dest.writestr(name,data)
    expect_failure(lambda:audit.read_bridge(badbridge,bindings),'INCIDENCE_HASH_MISMATCH')
    badbindings=json.loads(json.dumps(bindings));badbindings['selected_annual_profile_oracle'][0]['unique_issuer_party_year_edges']+=1
    common.write_json(fake_kit/'INPUT_BINDINGS.json',badbindings)
    badout=root/'bad_oracle';badout.mkdir()
    expect_failure(lambda:build.run(badout),'CONSERVATIVE_GRAPH_PROFILE_REPRODUCTION_FAILED')
    assert not (badout/'E7_PRIVATE_BRIDGE.zip').exists()
    common.write_json(fake_kit/'INPUT_BINDINGS.json',bindings)
    (raw/build.reader.VOLUMES[0]).write_bytes(b'changed source')
    badout=root/'bad_source';badout.mkdir()
    expect_failure(lambda:build.run(badout),'BOUND_SOURCE_IDENTITY_MISMATCH')
    assert not (badout/'E7_PRIVATE_BRIDGE.zip').exists()
    checks.append('Bridge corruption, annual reproduction mismatch and changed source block successful output')
    outside=root/'outside';outside.mkdir();link=root/'link';link.symlink_to(outside,target_is_directory=True)
    expect_failure(lambda:common.new_output(link/'run',root),'SYMLINK_PATH_REJECTED')
    expect_failure(lambda:common.new_output(root.parent/'escaped_run',root),'WRITE_BOUNDARY_REJECTED')
    checks.append('Output path rejects symlink components and boundary escape')
    assert not any('PRIVATE_' in x or x.endswith('.parquet') or x.endswith('.pt') for x in audit.RETURN_FILES)
    common.zip_exact(root/'aggregate.zip',out,['PRIVATE_BRIDGE_MANIFEST.json'])
    with zipfile.ZipFile(root/'aggregate.zip') as z:
        assert z.namelist()==['PRIVATE_BRIDGE_MANIFEST.json']
    checks.append('Aggregate return whitelist excludes private edges, support indices and model artifacts')
    build.KIT,build.PROJECT = saved_kit,saved_project;audit.KIT=saved_audit_kit

evidence = {'status':'PASS','checks':checks,'synthetic_only':True,
  'actual_tencent_source_build_run_here':False,'actual_4090_parquet_alignment_run_here':False,
  'actual_cuda_operator_tests_run_here':False,
  'local_limit':'No pyarrow or torch/torch_geometric installed; real Parquet/CUDA checks run on the supplied 4090 interpreter.'}
common.write_json(ROOT/'synthetic_validation.json',evidence)
print(json.dumps(evidence,ensure_ascii=False,indent=2))
