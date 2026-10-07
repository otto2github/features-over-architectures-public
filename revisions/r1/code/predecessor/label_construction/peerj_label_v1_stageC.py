#!/usr/bin/env python3
"""
PeerJ 141707 — Label v1 Stage C
Freeze CNINFO annual-report public anchors and integrate qualifying CSRC
public-sanction timing with Stage-A evidence.

This stage is evidence adjudication, NOT final label creation.

Inputs
------
- latest completed Stage A:
    pair_ledger_v1_stageA.parquet
    firm_year_support_v1_stageA.parquet
    manifest_v1_stageA.json
- latest completed Stage B3d:
    annual_report_public_anchors_CNINFO_v4.parquet
    manifest_v1_stageB3d.json
- pinned CSMAR violation descriptor:
    STK_Violation_Main[DES][xlsx].txt

Core timing rules
-----------------
1) Annual-report prediction anchor = CNINFO public annual-report announcement date.
2) CSMAR DeclareDate is used only as the PUBLIC sanction-announcement date because
   the pinned descriptor explicitly defines it as 公告日期.
3) CSMAR DisposalDate is retained separately as 处理文件日期.
4) A post-filing candidate requires DeclareDate > annual-report public anchor.
5) If public announcement is post-filing but DisposalDate <= anchor, this is a
   timing-conflict class, not an automatically accepted positive.
6) Missing anchor/date, raw-year-only evidence, and other unresolved evidence
   remain unresolved.
7) No final 0/1 labels are produced here.
8) No training is run.

Writes only under ~/peerj_141707_audit/label_v1_stageC/
On macOS, --reveal selects the final report in Finder.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

BASE = Path.home() / "peerj_141707_audit"
RAW_DIR = Path(os.environ.get("PEERJ_VIOLATION_SOURCE_DIR", "private_inputs/csmar/violation"))
DESCRIPTOR = RAW_DIR / "STK_Violation_Main[DES][xlsx].txt"
DESCRIPTOR_SHA256 = "14c4cd27901ee9e4ce629c869b529c8d7ae831df8157dc76d30733bebea55e88"
CUTOFF = date(2026, 5, 8)

PRE_SUPPORT = {"A_text_year_supported_pre_anchor", "B_raw_year_only_pre_anchor"}

def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def parse_date(x):
    if x is None:
        return None
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    s = str(x).strip()
    if not s or s.lower() in {"nan","none","nat"}:
        return None
    s = s.replace("年","-").replace("月","-").replace("日","").replace("/", "-").replace(".", "-")
    m = re.search(r"((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})", s)
    if not m:
        m = re.search(r"((?:19|20)\d{2})(\d{2})(\d{2})", re.sub(r"\D","",s))
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None

def quantiles(vals):
    vals = sorted(vals)
    if not vals:
        return {}
    def q(p):
        x = (len(vals)-1)*p
        lo = math.floor(x); hi = math.ceil(x)
        if lo == hi:
            return vals[lo]
        return vals[lo]*(hi-x)+vals[hi]*(x-lo)
    return {
        "n": len(vals),
        "min": vals[0],
        "p05": round(q(.05),1),
        "p25": round(q(.25),1),
        "median": round(q(.50),1),
        "p75": round(q(.75),1),
        "p95": round(q(.95),1),
        "max": vals[-1],
    }

def read_parquet(path):
    import pyarrow.parquet as pq
    return pq.read_table(path, use_threads=False).to_pylist()

def write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(list(rows)), path, compression="zstd")

def latest_completed(parent: Path, prefix: str, manifest_name: str):
    cand=[]
    if not parent.is_dir():
        raise SystemExit(f"MISSING_PARENT:{parent}")
    for d in parent.glob(prefix+"*"):
        m=d/manifest_name
        if not m.is_file():
            continue
        try:
            j=json.loads(m.read_text(encoding="utf-8"))
            if j.get("completed", True):
                cand.append((m.stat().st_mtime,d,j))
        except Exception:
            continue
    if not cand:
        raise SystemExit(f"NO_COMPLETED_RUN:{parent}/{prefix}*")
    _,d,j=max(cand,key=lambda x:x[0])
    return d,j

def descriptor_semantics():
    if sha256(DESCRIPTOR) != DESCRIPTOR_SHA256:
        raise ValueError("DESCRIPTOR_HASH_MISMATCH")
    txt = DESCRIPTOR.read_text(encoding="utf-8-sig", errors="strict")
    decl = None
    disp = None
    for line in txt.splitlines():
        if line.startswith("DeclareDate "):
            decl=line.strip()
        if line.startswith("DisposalDate "):
            disp=line.strip()
    if decl is None or disp is None:
        raise ValueError("DESCRIPTOR_DATE_FIELDS_NOT_FOUND")
    if "[公告日期]" not in decl:
        raise ValueError("DECLAREDATE_NOT_EXPLICITLY_PUBLIC_ANNOUNCEMENT_DATE")
    if "[处理文件日期]" not in disp:
        raise ValueError("DISPOSALDATE_SEMANTICS_UNEXPECTED")
    return decl,disp

def reveal(path):
    if sys.platform=="darwin":
        subprocess.run(["open","-R",str(path)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def classify_pair(p, anchor):
    pre = p.get("preanchor_status")
    if pre not in PRE_SUPPORT:
        return "not_preanchor_supported_candidate"

    if anchor is None:
        return "unresolved_missing_annual_report_public_anchor"

    declare = parse_date(p.get("declare_date"))
    disposal = parse_date(p.get("disposal_date"))

    if declare is None:
        return "unresolved_missing_public_sanction_announcement_date"

    if declare <= anchor:
        return "public_sanction_on_or_before_annual_report_anchor"

    # Public announcement is post-filing.
    if disposal is None:
        return "post_filing_public_supported_disposal_date_unresolved"
    if disposal <= anchor:
        return "timing_conflict_public_postfiling_but_document_date_pre_or_same"
    return "post_filing_public_and_document_supported"

def strength_rank(state):
    order = {
        "A_postfiling_public_and_document_supported": 100,
        "A_postfiling_public_supported_disposal_unresolved": 90,
        "B_postfiling_raw_year_only_public_and_document_supported": 80,
        "B_postfiling_raw_year_only_disposal_unresolved": 70,
        "timing_conflict_public_postfiling_but_document_date_pre_or_same": 60,
        "public_sanction_on_or_before_annual_report_anchor": 50,
        "unresolved_missing_public_sanction_announcement_date": 40,
        "unresolved_missing_annual_report_public_anchor": 30,
        "unresolved_preanchor_evidence": 20,
        "no_machine_support_not_negative": 10,
    }
    return order.get(state,0)

def aggregate_state(pre_state, pair_states):
    ps=set(pair_states)

    if pre_state == "A_text_year_supported_pre_anchor":
        if "post_filing_public_and_document_supported" in ps:
            return "A_postfiling_public_and_document_supported"
        if "post_filing_public_supported_disposal_date_unresolved" in ps:
            return "A_postfiling_public_supported_disposal_unresolved"
    elif pre_state == "B_raw_year_only_pre_anchor":
        if "post_filing_public_and_document_supported" in ps:
            return "B_postfiling_raw_year_only_public_and_document_supported"
        if "post_filing_public_supported_disposal_date_unresolved" in ps:
            return "B_postfiling_raw_year_only_disposal_unresolved"

    if "timing_conflict_public_postfiling_but_document_date_pre_or_same" in ps:
        return "timing_conflict_public_postfiling_but_document_date_pre_or_same"
    if "public_sanction_on_or_before_annual_report_anchor" in ps:
        return "public_sanction_on_or_before_annual_report_anchor"
    if "unresolved_missing_public_sanction_announcement_date" in ps:
        return "unresolved_missing_public_sanction_announcement_date"
    if "unresolved_missing_annual_report_public_anchor" in ps:
        return "unresolved_missing_annual_report_public_anchor"
    if pre_state == "unresolved_pre_anchor":
        return "unresolved_preanchor_evidence"
    return "no_machine_support_not_negative"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage-c", action="store_true")
    ap.add_argument("--reveal", action="store_true")
    args=ap.parse_args()
    if not args.stage_c:
        ap.error("--stage-c required")

    os.umask(0o077)
    outroot=BASE/"label_v1_stageC"
    outroot.mkdir(parents=True,exist_ok=True,mode=0o700)
    out=outroot/("stageC_"+datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8])
    out.mkdir(mode=0o700)
    report=out/"label_v1_stageC_report.txt"

    try:
        print("[1/6] 核验 Stage A、Stage B3d 与 CSMAR 日期字段语义……",flush=True)
        stageA,maniA=latest_completed(BASE/"label_v1_stageA","stageA_","manifest_v1_stageA.json")
        stageB,maniB=latest_completed(BASE/"label_v1_stageB3d","stageB3d_","manifest_v1_stageB3d.json")
        decl_sem,disp_sem=descriptor_semantics()

        anchor_file=stageB/"annual_report_public_anchors_CNINFO_v4.parquet"
        if not anchor_file.is_file():
            raise ValueError("B3D_ANCHOR_FILE_MISSING")
        if maniB.get("anchor_sha256") and sha256(anchor_file)!=maniB["anchor_sha256"]:
            raise ValueError("B3D_ANCHOR_HASH_MISMATCH")
        if float(maniB.get("coverage",0)) < .95:
            raise ValueError("B3D_COVERAGE_NOT_FROZEN_ELIGIBLE")

        pair_file=stageA/"pair_ledger_v1_stageA.parquet"
        fy_file=stageA/"firm_year_support_v1_stageA.parquet"
        if not pair_file.is_file() or not fy_file.is_file():
            raise ValueError("STAGE_A_LEDGER_MISSING")

        print("[2/6] 冻结 49k+ CNINFO 年报公开日期，并接入 DeclareDate/DisposalDate……",flush=True)
        anchors=read_parquet(anchor_file)
        amap={}
        anchor_follow=defaultdict(list)
        anchor_missing=Counter()
        for r in anchors:
            key=(r["company_code"],int(r["fiscal_year"]))
            d=parse_date(r.get("annual_report_public_anchor"))
            amap[key]=d
            if d:
                anchor_follow[key[1]].append((CUTOFF-d).days)
            else:
                anchor_missing[key[1]]+=1

        pairs=read_parquet(pair_file)
        pair_out=[]
        pair_counts={"strict":Counter(),"loose":Counter()}
        pair_states_by_key_scope=defaultdict(list)
        earliest_decl_by_key_scope=defaultdict(list)
        earliest_disp_by_key_scope=defaultdict(list)

        for p in pairs:
            scope=p["scope"]
            year=p.get("raw_violation_year")
            try:
                year=int(year) if year is not None else None
            except Exception:
                year=None
            key=(p.get("company_code"),year) if p.get("company_code") and year is not None else None
            anchor=amap.get(key) if key else None
            state=classify_pair(p,anchor)
            pair_counts[scope][state]+=1

            decl=parse_date(p.get("declare_date"))
            disp=parse_date(p.get("disposal_date"))
            if key and p.get("preanchor_status") in PRE_SUPPORT:
                pair_states_by_key_scope[(scope,key)].append(state)
                if decl: earliest_decl_by_key_scope[(scope,key)].append(decl)
                if disp: earliest_disp_by_key_scope[(scope,key)].append(disp)

            q=dict(p)
            q["annual_report_public_anchor"]=anchor.isoformat() if anchor else None
            q["public_sanction_announcement_date"]=decl.isoformat() if decl else None
            q["sanction_document_date"]=disp.isoformat() if disp else None
            q["stageC_timing_state"]=state
            q["final_label"]=None
            q["training_ready"]=False
            pair_out.append(q)

        write_parquet(out/"pair_timing_v1_stageC.parquet",pair_out)

        print("[3/6] 聚合至 51,675×2 firm-year/scope，重新对照 legacy……",flush=True)
        fyr=read_parquet(fy_file)
        fy_out=[]
        state_counts={"strict":Counter(),"loose":Counter()}
        legacy_counts={"strict":Counter(),"loose":Counter()}
        partition_counts={"strict":defaultdict(Counter),"loose":defaultdict(Counter)}
        review=[]

        accepted_A = "A_postfiling_public_and_document_supported"
        accepted_A_soft = "A_postfiling_public_supported_disposal_unresolved"
        accepted_B = "B_postfiling_raw_year_only_public_and_document_supported"
        accepted_B_soft = "B_postfiling_raw_year_only_disposal_unresolved"

        for r in fyr:
            scope=r["scope"]
            key=(r["company_code"],int(r["fiscal_year"]))
            anchor=amap.get(key)
            states=pair_states_by_key_scope.get((scope,key),[])
            state=aggregate_state(r["preanchor_evidence_state"],states)
            state_counts[scope][state]+=1
            part=r["partition"]
            partition_counts[scope][part][state]+=1

            legacy=int(r["legacy_label"])
            legacy_counts[scope][f"legacy{legacy}__{state}"]+=1

            decls=earliest_decl_by_key_scope.get((scope,key),[])
            disps=earliest_disp_by_key_scope.get((scope,key),[])
            earliest_decl=min(decls) if decls else None
            earliest_disp=min(disps) if disps else None
            follow=(CUTOFF-anchor).days if anchor else None

            o=dict(r)
            o["annual_report_public_anchor"]=anchor.isoformat() if anchor else None
            o["earliest_candidate_public_sanction_date"]=earliest_decl.isoformat() if earliest_decl else None
            o["earliest_candidate_document_date"]=earliest_disp.isoformat() if earliest_disp else None
            o["followup_days_to_2026_05_08"]=follow
            o["stageC_evidence_state"]=state
            o["final_label"]=None
            o["training_ready"]=False
            fy_out.append(o)

            needs_review = (
                state not in {accepted_A, "no_machine_support_not_negative"} or
                (legacy==1 and state!=accepted_A) or
                (legacy==0 and state in {accepted_A,accepted_A_soft,accepted_B,accepted_B_soft})
            )
            if needs_review:
                review.append({
                    "company_code":"'"+key[0],
                    "fiscal_year":key[1],
                    "partition":part,
                    "scope":scope,
                    "legacy_label":legacy,
                    "stageA_preanchor_state":r["preanchor_evidence_state"],
                    "stageC_evidence_state":state,
                    "annual_report_public_anchor":o["annual_report_public_anchor"],
                    "earliest_candidate_public_sanction_date":o["earliest_candidate_public_sanction_date"],
                    "earliest_candidate_document_date":o["earliest_candidate_document_date"],
                    "followup_days_to_2026_05_08":follow,
                })

        write_parquet(out/"firm_year_evidence_v1_stageC.parquet",fy_out)

        with (out/"review_queue_STAGEC_LOCAL_ONLY.csv").open("w",encoding="utf-8",newline="") as f:
            fields=list(review[0]) if review else ["company_code","fiscal_year","scope"]
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(review)

        print("[4/6] 量化 follow-up maturity，不先拍板主测试窗口……",flush=True)
        maturity={}
        for y in range(2010,2025):
            vals=anchor_follow.get(y,[])
            maturity[str(y)]={
                **quantiles(vals),
                "panel_missing_anchor":anchor_missing[y],
                "ge_365":sum(v>=365 for v in vals),
                "ge_730":sum(v>=730 for v in vals),
                "ge_1095":sum(v>=1095 for v in vals),
                "ge_1460":sum(v>=1460 for v in vals),
            }

        print("[5/6] 生成提交前关键差异统计；仍不赋最终0/1……",flush=True)
        report_lines=[
            "PeerJ 141707 | Label v1 Stage C post-filing evidence integration",
            "NO FINAL LABELS; NO TRAINING",
            "Local time: "+datetime.now().astimezone().isoformat(),
            "Stage A: "+str(stageA),
            "Stage B3d: "+str(stageB),
            "Output: "+str(out),"",
            "A. FROZEN SOURCE SEMANTICS",
            "annual_report_anchor=CNINFO first public full annual-report announcement date",
            "DeclareDate_semantics="+decl_sem,
            "DisposalDate_semantics="+disp_sem,
            "post_filing_public_rule=DeclareDate strictly later than CNINFO annual-report public anchor",
            "document_date_rule=DisposalDate retained separately; pre/same-anchor document date creates timing-conflict class",
            "FIN_Audit_Annodt_used=false","",
            "B. ANNUAL-REPORT ANCHOR",
            f"anchor_rows={len(amap)}",
            f"anchor_nonmissing={sum(v is not None for v in amap.values())}",
            f"anchor_missing={sum(v is None for v in amap.values())}",
            "anchor_sha256="+sha256(anchor_file),"",
            "C. PAIR-LEVEL TIMING STATUS",
            "strict="+json.dumps(dict(pair_counts["strict"]),ensure_ascii=False,sort_keys=True),
            "loose="+json.dumps(dict(pair_counts["loose"]),ensure_ascii=False,sort_keys=True),"",
            "D. FIRM-YEAR EVIDENCE STATUS",
            "strict="+json.dumps(dict(state_counts["strict"]),ensure_ascii=False,sort_keys=True),
            "loose="+json.dumps(dict(state_counts["loose"]),ensure_ascii=False,sort_keys=True),"",
            "E. PARTITION BREAKDOWN",
            "strict="+json.dumps({k:dict(v) for k,v in partition_counts["strict"].items()},ensure_ascii=False,sort_keys=True),
            "loose="+json.dumps({k:dict(v) for k,v in partition_counts["loose"].items()},ensure_ascii=False,sort_keys=True),"",
            "F. LEGACY vs STAGE-C EVIDENCE (NOT ERROR ADJUDICATION)",
            "strict="+json.dumps(dict(legacy_counts["strict"]),ensure_ascii=False,sort_keys=True),
            "loose="+json.dumps(dict(legacy_counts["loose"]),ensure_ascii=False,sort_keys=True),"",
            "G. FOLLOW-UP MATURITY TO FIXED CUTOFF 2026-05-08",
            json.dumps(maturity,ensure_ascii=False,sort_keys=True),"",
            "H. REVIEW WORKLOAD",
            f"review_queue_firm_year_scope_rows={len(review)}",
            "Review queue is targeted adjudication; it is not an estimated error rate.","",
            "I. NEXT GATE",
            "Stage C intentionally does not turn no-observed-sanction rows into negatives.",
            "Next: freeze maturity rule + adjudicate A/B/raw-year conflicts + produce Label v1.0 and legacy diff.",
            "",
            "source_files_modified=false",
            "training_or_prediction_run=false",
            "final_nonnull_labels_written=0",
            "STATUS: LABEL_V1_STAGE_C_COMPLETE_READY_FOR_MATURITY_AND_LABEL_FREEZE",
            "Upload only label_v1_stageC_report.txt.",
        ]
        report.write_text("\n".join(report_lines)+"\n",encoding="utf-8")

        print("[6/6] 写出 manifest 并在 Finder 定位报告……",flush=True)
        manifest={
            "completed":True,
            "created_at":datetime.now().astimezone().isoformat(),
            "stageA_manifest_sha256":sha256(stageA/"manifest_v1_stageA.json"),
            "stageB3d_manifest_sha256":sha256(stageB/"manifest_v1_stageB3d.json"),
            "anchor_sha256":sha256(anchor_file),
            "descriptor_sha256":DESCRIPTOR_SHA256,
            "final_labels_authorized":False,
            "training_authorized":False,
        }
        (out/"manifest_v1_stageC.json").write_text(
            json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")

        print("STATUS: LABEL_V1_STAGE_C_COMPLETE_READY_FOR_MATURITY_AND_LABEL_FREEZE")
        print("请只上传：",report)
        if args.reveal:
            reveal(report)

    except Exception as e:
        report.write_text(
            "PeerJ 141707 | Stage C FAILED SAFELY\n"
            f"ERROR_TYPE={type(e).__name__}\nERROR={e}\n"
            "source_files_modified=false\ntraining_or_prediction_run=false\nfinal_nonnull_labels_written=0\n"
            "STATUS: LABEL_V1_STAGE_C_FAILED_SAFE\n",
            encoding="utf-8"
        )
        if args.reveal:
            reveal(report)
        raise

if __name__=="__main__":
    main()
