#!/usr/bin/env python3
"""
PeerJ 141707 — Label v1.0 Stage G4
Entity-level graph preflight + isolated-company mapping repair.

Why G3 failed
-------------
node_features_v1_1 is a firm-year table, but node_id is intentionally ENTITY-level:
the same listed company node (e.g. C:000001.SZ) appears once per fiscal year.
Therefore node_id repetition across years is expected and must NOT be treated
as a duplicate-node error.

Correct mapping:
    firm-year row
      -> stable company node_id in node_features_v1_1
      -> entity-level node_id_index.node_id
      -> node_idx
Each year-specific graph writes that year's feature vector onto the same
entity-level company node_idx.

This stage:
- verifies the frozen Label v1.0 and both feature panels;
- verifies node_id -> firm identity consistency across years;
- joins unique company entity node_ids to node_id_index;
- if only a tiny number of company entity nodes are absent, creates a derived
  mapping by appending those entity nodes with new indices and NO invented edges;
- audits global_edge_index using its actual schema:
      src_idx, dst_idx, edge_type, year
- does NOT train models, use GPU, or overwrite source files.

Writes only under ~/peerj_141707_audit/label_v1_stageG4/
--reveal opens Finder and selects the final report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

BASE = Path.home() / "peerj_141707_audit"
PROJECT = Path(os.environ.get("PEERJ_SOURCE_PROJECT_ROOT", "private_inputs/project"))

FEATURES = PROJECT / "data/processed/features/features_v1_1.parquet"
NODE_FEATURES = PROJECT / "data/processed/kg/node_features_v1_1.parquet"
NODE_INDEX = PROJECT / "data/processed/kg/node_id_index.parquet"
EDGES = PROJECT / "data/processed/kg/global_edge_index.parquet"

EXPECTED_LABEL_SHA = "1d50378138dfc65def76555c0a48aa9cea2656af4d456fe5178d9cf7eae31836"
EXPECTED_PANEL_ROWS = 51675
EXPECTED_PRIMARY_NEG = 39536
EXPECTED_PRIMARY_POS = 314
EXPECTED_OLD_NODE_ROWS = 303771
EXPECTED_EDGE_ROWS = 4469735
EXPECTED_RELATIONS = {"E1","E2","E3","E4","E5"}
MAX_AUTO_REPAIR_MISSING_ENTITY_NODES = 20

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def latest(parent: Path, prefix: str, manifest: str):
    cand=[]
    for d in parent.glob(prefix+"*"):
        m=d/manifest
        if not m.is_file():
            continue
        try:
            j=json.loads(m.read_text(encoding="utf-8"))
            if j.get("completed",True):
                cand.append((m.stat().st_mtime,d,j))
        except Exception:
            pass
    if not cand:
        raise SystemExit(f"NO_COMPLETED_RUN:{parent}/{prefix}*")
    _,d,j=max(cand,key=lambda x:x[0])
    return d,j

def read_table(path: Path, columns=None):
    import pyarrow.parquet as pq
    return pq.read_table(path,columns=columns,use_threads=False)

def read_rows(path: Path, columns=None):
    return read_table(path,columns).to_pylist()

def write_table(path: Path, table):
    import pyarrow.parquet as pq
    pq.write_table(table,path,compression="zstd")

def norm_firm(x):
    if x is None:return None
    s=str(x).strip()
    m=re.search(r"(?<!\d)(\d{6})(?!\d)",s)
    if m:return m.group(1)
    if re.fullmatch(r"\d+(?:\.0+)?",s):
        try:return f"{int(float(s)):06d}"[-6:]
        except Exception:pass
    return s

def canonical_rel(x):
    s=str(x).strip().upper()
    m=re.search(r"(?:^|[^A-Z0-9])E([1-5])(?:$|[^A-Z0-9])",s)
    if m:return "E"+m.group(1)
    if re.fullmatch(r"[1-5](?:\.0+)?",s):
        return "E"+str(int(float(s)))
    return s

def reveal(path: Path):
    if sys.platform=="darwin":
        subprocess.run(["open","-R",str(path)],
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage-g4",action="store_true")
    ap.add_argument("--reveal",action="store_true")
    args=ap.parse_args()
    if not args.stage_g4:
        ap.error("--stage-g4 required")

    os.umask(0o077)
    outroot=BASE/"label_v1_stageG4"
    outroot.mkdir(parents=True,exist_ok=True,mode=0o700)
    out=outroot/("stageG4_"+datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8])
    out.mkdir(mode=0o700)
    report=out/"label_v1_stageG4_report.txt"

    try:
        print("[1/8] 核验冻结 Label v1.0 与主计数……",flush=True)
        stageF,maniF=latest(BASE/"label_v1_stageF","stageF_","manifest_v1_stageF.json")
        labelfile=stageF/"fraud_labels_v1_0.parquet"
        if sha256(labelfile)!=EXPECTED_LABEL_SHA:
            raise ValueError("LABEL_HASH_MISMATCH")
        labels=read_rows(labelfile)
        if len(labels)!=EXPECTED_PANEL_ROWS:
            raise ValueError(f"LABEL_ROW_COUNT_MISMATCH:{len(labels)}")
        label_keys={(r["company_code"],int(r["fiscal_year"])) for r in labels}
        if len(label_keys)!=EXPECTED_PANEL_ROWS:
            raise ValueError("LABEL_KEYS_NOT_UNIQUE")
        npos=sum(r["label_v1_strict_ab_primary"]==1 for r in labels)
        nneg=sum(r["label_v1_strict_ab_primary"]==0 for r in labels)
        if (nneg,npos)!=(EXPECTED_PRIMARY_NEG,EXPECTED_PRIMARY_POS):
            raise ValueError(f"PRIMARY_COUNTS_MISMATCH:{nneg},{npos}")

        for p in (FEATURES,NODE_FEATURES,NODE_INDEX,EDGES):
            if not p.is_file():
                raise ValueError(f"REQUIRED_FILE_MISSING:{p}")

        print("[2/8] 核验 features / node_features 的51,675 firm-year键……",flush=True)
        frows=read_rows(FEATURES,["firm_id","year"])
        fkeys={(norm_firm(r["firm_id"]),int(r["year"])) for r in frows}
        if len(frows)!=EXPECTED_PANEL_ROWS or fkeys!=label_keys:
            raise ValueError("FEATURES_FIRMYEAR_KEY_MISMATCH")

        nfrows=read_rows(NODE_FEATURES,["node_id","ts_code","firm_id","year"])
        nf_by_key={}
        node_identity=defaultdict(set)
        years_by_node=defaultdict(set)
        for r in nfrows:
            k=(norm_firm(r["firm_id"]),int(r["year"]))
            if k in nf_by_key:
                raise ValueError(f"NODE_FEATURE_DUP_FIRMYEAR:{k}")
            nf_by_key[k]=r
            nid=str(r["node_id"])
            node_identity[nid].add((norm_firm(r["firm_id"]),str(r["ts_code"])))
            years_by_node[nid].add(int(r["year"]))

        if len(nfrows)!=EXPECTED_PANEL_ROWS or set(nf_by_key)!=label_keys:
            raise ValueError("NODE_FEATURES_FIRMYEAR_KEY_MISMATCH")

        unstable={nid:list(vals) for nid,vals in node_identity.items() if len(vals)!=1}
        if unstable:
            ex=list(unstable.items())[:10]
            raise ValueError(f"NODE_ID_MAPS_TO_MULTIPLE_FIRMS:{ex}")

        unique_company_nodes=set(node_identity)
        repeated_node_ids=sum(len(ys)>1 for ys in years_by_node.values())
        print(f"      unique company entity node_ids={len(unique_company_nodes)}; "
              f"repeated across >1 year={repeated_node_ids}",flush=True)

        print("[3/8] 核验 entity-level node_id_index，并定位缺失公司实体节点……",flush=True)
        import pyarrow as pa
        import pyarrow.compute as pc

        map_table=read_table(NODE_INDEX,["node_idx","node_id","node_type","ts_code"])
        if map_table.num_rows!=EXPECTED_OLD_NODE_ROWS:
            raise ValueError(f"OLD_NODE_INDEX_ROWCOUNT_UNEXPECTED:{map_table.num_rows}")

        md=map_table.to_pydict()
        if len(set(md["node_id"]))!=map_table.num_rows:
            raise ValueError("NODE_INDEX_NODE_ID_NOT_UNIQUE")
        if len(set(md["node_idx"]))!=map_table.num_rows:
            raise ValueError("NODE_INDEX_NODE_IDX_NOT_UNIQUE")
        idx_min=min(md["node_idx"]); idx_max=max(md["node_idx"])
        dense=(idx_min==0 and idx_max==map_table.num_rows-1 and
               set(md["node_idx"])==set(range(map_table.num_rows)))
        if not dense:
            raise ValueError(
                f"NODE_INDEX_NOT_DENSE_0_BASED:min={idx_min},max={idx_max},n={map_table.num_rows}"
            )

        map_by_nid={str(nid):(idx,nt,ts)
                    for nid,idx,nt,ts in zip(md["node_id"],md["node_idx"],md["node_type"],md["ts_code"])}

        mapped_feature_nodes=unique_company_nodes & set(map_by_nid)
        missing_entity_nodes=sorted(unique_company_nodes-set(map_by_nid))

        # Infer the company node_type from mapped company entities.
        company_type_counts=Counter(str(map_by_nid[nid][1]) for nid in mapped_feature_nodes)
        if not company_type_counts:
            raise ValueError("NO_FEATURE_COMPANY_NODES_FOUND_IN_NODE_INDEX")
        company_type,ctype_n=company_type_counts.most_common(1)[0]
        if ctype_n < 0.99*sum(company_type_counts.values()):
            raise ValueError(f"COMPANY_NODE_TYPE_UNSTABLE:{dict(company_type_counts)}")

        missing_firmyears=[]
        missing_primary=[]
        missing_primary_split=Counter()
        missing_primary_label=Counter()
        for r in labels:
            k=(r["company_code"],int(r["fiscal_year"]))
            nid=str(nf_by_key[k]["node_id"])
            if nid not in map_by_nid:
                x={
                    "company_code":k[0],"fiscal_year":k[1],
                    "node_id":nid,"ts_code":str(nf_by_key[k]["ts_code"]),
                    "split_v1":r["split_v1"],
                    "primary_label":r["label_v1_strict_ab_primary"],
                    "freeze_status":r["freeze_status"],
                }
                missing_firmyears.append(x)
                if r["label_v1_strict_ab_primary"] is not None:
                    missing_primary.append(x)
                    missing_primary_split[r["split_v1"]]+=1
                    missing_primary_label[str(r["label_v1_strict_ab_primary"])]+=1

        print(f"      old mapping company entities present={len(mapped_feature_nodes)}; "
              f"missing entity nodes={len(missing_entity_nodes)}; "
              f"affected firm-years={len(missing_firmyears)}; "
              f"affected primary-labeled={len(missing_primary)}",flush=True)

        print("[4/8] 若缺失实体极少，生成派生mapping；旧indices完全不动，不添加任何边……",flush=True)
        derived_mapping=None
        appended=[]
        repair_status="not_needed"

        if missing_entity_nodes:
            if len(missing_entity_nodes)>MAX_AUTO_REPAIR_MISSING_ENTITY_NODES:
                repair_status="blocked_too_many_missing_entity_nodes"
            else:
                repair_status="append_missing_company_entities_as_isolated_nodes"
                # deterministic sort by node_id
                new_rows=[]
                next_idx=map_table.num_rows
                for nid in missing_entity_nodes:
                    # identity is guaranteed one-to-one above
                    firm,ts=node_identity[nid].copy().pop()
                    new_rows.append({
                        "node_idx":next_idx,
                        "node_id":nid,
                        "node_type":company_type,
                        "ts_code":ts,
                    })
                    appended.append({
                        "node_idx":next_idx,
                        "node_id":nid,
                        "firm_id":"'"+firm,
                        "ts_code":ts,
                        "years":";".join(str(y) for y in sorted(years_by_node[nid])),
                    })
                    next_idx+=1

                add=pa.Table.from_pylist(new_rows,schema=map_table.schema)
                repaired=pa.concat_tables([map_table,add])
                derived_mapping=out/"node_id_index_label_v1_0_fiverel_derived.parquet"
                write_table(derived_mapping,repaired)

                rd=repaired.to_pydict()
                if rd["node_idx"][:map_table.num_rows]!=md["node_idx"]:
                    raise ValueError("REPAIR_CHANGED_OLD_NODE_IDX")
                if rd["node_id"][:map_table.num_rows]!=md["node_id"]:
                    raise ValueError("REPAIR_CHANGED_OLD_NODE_ID")
                if len(set(rd["node_id"]))!=repaired.num_rows:
                    raise ValueError("REPAIR_DUP_NODE_ID")
                if len(set(rd["node_idx"]))!=repaired.num_rows:
                    raise ValueError("REPAIR_DUP_NODE_IDX")

        if appended:
            with (out/"appended_company_entities_STAGEG4_LOCAL_ONLY.csv").open(
                    "w",encoding="utf-8",newline="") as f:
                fields=["node_idx","node_id","firm_id","ts_code","years"]
                w=csv.DictWriter(f,fieldnames=fields)
                w.writeheader();w.writerows(appended)

        print("[5/8] 按实际schema核验E1–E5 edge artifact……",flush=True)
        edge_table=read_table(EDGES,["src_idx","dst_idx","edge_type","year"])
        if edge_table.num_rows!=EXPECTED_EDGE_ROWS:
            raise ValueError(f"EDGE_ROWCOUNT_UNEXPECTED:{edge_table.num_rows}")

        ed=edge_table.to_pydict()
        raw_rel=Counter(str(x) for x in ed["edge_type"])
        canon_rel=Counter(canonical_rel(x) for x in ed["edge_type"])
        relset=set(canon_rel)

        src=edge_table["src_idx"]
        dst=edge_table["dst_idx"]
        srcmm=pc.min_max(src).as_py()
        dstmm=pc.min_max(dst).as_py()
        min_endpoint=min(srcmm["min"],dstmm["min"])
        max_endpoint=max(srcmm["max"],dstmm["max"])
        selfloops=int(pc.sum(pc.cast(pc.equal(src,dst),pa.int64())).as_py() or 0)
        years=[int(x) for x in ed["year"] if x is not None]
        yr_min=min(years) if years else None
        yr_max=max(years) if years else None
        endpoints_valid=(min_endpoint>=0 and max_endpoint<map_table.num_rows)

        print(f"      edges={edge_table.num_rows}; relations={dict(canon_rel)}; "
              f"selfloops={selfloops}; endpoint=[{min_endpoint},{max_endpoint}]; "
              f"year=[{yr_min},{yr_max}]",flush=True)

        print("[6/8] 检查派生mapping后的panel/primary覆盖……",flush=True)
        final_map_ids=set(map_by_nid)|set(missing_entity_nodes if derived_mapping else [])
        final_missing_panel=[]
        final_missing_primary=[]
        for r in labels:
            k=(r["company_code"],int(r["fiscal_year"]))
            nid=str(nf_by_key[k]["node_id"])
            if nid not in final_map_ids:
                final_missing_panel.append(k)
                if r["label_v1_strict_ab_primary"] is not None:
                    final_missing_primary.append(k)

        blockers=[]
        if repair_status=="blocked_too_many_missing_entity_nodes":
            blockers.append(f"missing_entity_nodes={len(missing_entity_nodes)}_exceeds_auto_repair_limit")
        if relset!=EXPECTED_RELATIONS:
            blockers.append(f"relation_set_not_exact_E1_E5:{sorted(relset)}")
        if selfloops!=0:
            blockers.append(f"unexpected_selfloops={selfloops}")
        if not endpoints_valid:
            blockers.append(
                f"edge_endpoint_outside_old_mapping:min={min_endpoint},max={max_endpoint},n={map_table.num_rows}"
            )
        if final_missing_panel:
            blockers.append(f"panel_firmyears_still_unmapped={len(final_missing_panel)}")
        if final_missing_primary:
            blockers.append(f"primary_firmyears_still_unmapped={len(final_missing_primary)}")

        ready=not blockers

        print("[7/8] 计算输入hash并形成正式rerun mapping选择……",flush=True)
        label_sha=sha256(labelfile)
        features_sha=sha256(FEATURES)
        node_features_sha=sha256(NODE_FEATURES)
        old_mapping_sha=sha256(NODE_INDEX)
        derived_sha=sha256(derived_mapping) if derived_mapping else None
        edge_sha=sha256(EDGES)

        mapping_for_rerun=derived_mapping if derived_mapping else NODE_INDEX

        print("[8/8] 写出G4报告并Finder定位……",flush=True)
        lines=[
            "PeerJ 141707 | Label v1.0 Stage G4 entity-level graph preflight",
            "NO MODEL TRAINING; NO GPU",
            "Local time: "+datetime.now().astimezone().isoformat(),
            "Stage F: "+str(stageF),
            "Output: "+str(out),"",
            "A. ENTITY-LEVEL NODE SEMANTICS",
            "node_features_unit=firm-year rows",
            "node_id_unit=stable listed-company entity",
            "same_node_id_across_years_is_expected=true",
            f"node_feature_rows={len(nfrows)}",
            f"unique_company_entity_node_ids={len(unique_company_nodes)}",
            f"company_entity_node_ids_repeated_across_multiple_years={repeated_node_ids}",
            "node_id_identity_consistency_passed=true","",
            "B. FROZEN LABEL / FEATURE PANEL",
            f"label_rows={len(labels)}",
            f"primary_positive_n={npos}",
            f"primary_negative_n={nneg}",
            "features_firmyear_exact_match=true",
            "node_features_firmyear_exact_match=true",
            "label_sha256="+label_sha,
            "features_sha256="+features_sha,
            "node_features_sha256="+node_features_sha,"",
            "C. OLD ENTITY NODE MAPPING",
            "node_index_path="+str(NODE_INDEX),
            "node_index_sha256="+old_mapping_sha,
            f"node_index_rows={map_table.num_rows}",
            f"dense_zero_based={str(dense).lower()}",
            "company_node_type_counts="+json.dumps(dict(company_type_counts),ensure_ascii=False,sort_keys=True),
            f"feature_company_entities_present_in_old_mapping={len(mapped_feature_nodes)}",
            f"missing_company_entity_nodes={len(missing_entity_nodes)}",
            f"affected_panel_firmyears={len(missing_firmyears)}",
            f"affected_primary_labeled_firmyears={len(missing_primary)}",
            "affected_primary_by_split="+json.dumps(dict(missing_primary_split),ensure_ascii=False,sort_keys=True),
            "affected_primary_by_label="+json.dumps(dict(missing_primary_label),ensure_ascii=False,sort_keys=True),"",
            "D. DERIVED MAPPING REPAIR",
            "repair_status="+repair_status,
            "repair_rule=append only missing stable company entity node_ids after old max node_idx; preserve every old index; invent zero graph edges",
            "derived_mapping_path="+(str(derived_mapping) if derived_mapping else "NONE"),
            "derived_mapping_sha256="+(derived_sha or "NONE"),
            f"appended_entity_nodes={len(appended)}",
            f"final_unmapped_panel_firmyears={len(final_missing_panel)}",
            f"final_unmapped_primary_firmyears={len(final_missing_primary)}",
            "mapping_for_formal_rerun="+str(mapping_for_rerun),"",
            "E. FIVE-RELATION EDGE ARTIFACT",
            "edge_path="+str(EDGES),
            "edge_sha256="+edge_sha,
            f"edge_rows={edge_table.num_rows}",
            "raw_relation_counts="+json.dumps(dict(raw_rel),ensure_ascii=False,sort_keys=True),
            "canonical_relation_counts="+json.dumps(dict(canon_rel),ensure_ascii=False,sort_keys=True),
            "canonical_relation_set="+json.dumps(sorted(relset),ensure_ascii=False),
            f"selfloops={selfloops}",
            f"endpoint_min={min_endpoint}",
            f"endpoint_max={max_endpoint}",
            f"endpoints_valid_against_old_mapping={str(endpoints_valid).lower()}",
            f"edge_year_min={yr_min}",
            f"edge_year_max={yr_max}","",
            "F. DECISION",
            "formal_rerun_code_freeze_ready="+str(ready).lower(),
            "blockers="+json.dumps(blockers,ensure_ascii=False),
        ]

        if ready:
            lines += [
                "The entity-level graph interpretation is consistent with the firm-year feature panel.",
                "If a derived mapping was created, use it for formal reruns; appended company entities remain isolated because no source E1-E5 edge exists for them.",
                "STATUS: LABEL_V1_STAGE_G4_PASS_READY_FOR_RERUN_CODE_FREEZE",
            ]
        else:
            lines += [
                "Do not start GPU training until listed blockers are resolved.",
                "STATUS: LABEL_V1_STAGE_G4_STOP_GRAPH_INPUT_PREFLIGHT",
            ]

        lines += [
            "",
            "source_files_modified=false",
            "model_training_run=false",
            "Upload only label_v1_stageG4_report.txt.",
        ]
        report.write_text("\n".join(lines)+"\n",encoding="utf-8")

        (out/"manifest_v1_stageG4.json").write_text(json.dumps({
            "completed":True,
            "created_at":datetime.now().astimezone().isoformat(),
            "stageF_manifest_sha256":sha256(stageF/"manifest_v1_stageF.json"),
            "label_sha256":label_sha,
            "features_sha256":features_sha,
            "node_features_sha256":node_features_sha,
            "old_node_index_sha256":old_mapping_sha,
            "derived_mapping_path":str(derived_mapping) if derived_mapping else None,
            "derived_mapping_sha256":derived_sha,
            "appended_entity_nodes":len(appended),
            "edge_sha256":edge_sha,
            "mapping_for_formal_rerun":str(mapping_for_rerun),
            "formal_rerun_code_freeze_ready":ready,
            "training_authorized":False,
        },ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")

        print([x for x in lines if x.startswith("STATUS:")][0])
        print("请只上传：",report)
        if args.reveal:
            reveal(report)

    except Exception as e:
        report.write_text(
            "PeerJ 141707 | Stage G4 FAILED SAFELY\n"
            f"ERROR_TYPE={type(e).__name__}\nERROR={e}\n"
            "source_files_modified=false\nmodel_training_run=false\n"
            "STATUS: LABEL_V1_STAGE_G4_FAILED_SAFE\n",
            encoding="utf-8"
        )
        if args.reveal:
            reveal(report)
        raise

if __name__=="__main__":
    main()
