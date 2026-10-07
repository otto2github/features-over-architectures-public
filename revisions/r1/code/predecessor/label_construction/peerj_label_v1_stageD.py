#!/usr/bin/env python3
"""
PeerJ 141707 — Label v1 Stage D
Legacy-source decomposition and temporal audit before Label v1.0 freeze.

Why this stage
--------------
Stage C solved the annual-report public-date anchor, but its machine-supported
CSMAR evidence covers only a small fraction of legacy positives. The historical
strict label is a union that includes the older v0.7 source. Therefore
"legacy positive without Stage-C machine support" must NOT be interpreted as
an error.

Stage D quantifies the source decomposition exactly and tests the historical
v0.7 notice-date proxy against the now-frozen CNINFO annual-report public
anchor.

Reads:
  - fraud_labels_v0_8.parquet
  - fraud_labels_v0.parquet
  - latest Stage C firm_year_evidence_v1_stageC.parquet
  - latest Stage B3d annual_report_public_anchors_CNINFO_v4.parquet

Writes only under:
  ~/peerj_141707_audit/label_v1_stageD/

No final labels. No training. No source overwrite. No network.
--reveal opens Finder and selects the final report.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime
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
LABEL_V08 = PROJECT / "data/processed/labels/fraud_labels_v0_8.parquet"
LABEL_V0 = PROJECT / "data/processed/labels/fraud_labels_v0.parquet"

V08_SHA256 = "af99c86bd452f54d7e79990e9fd9c6b804cf1ac45d6b6b7e81eba98764a724ac"
V0_SHA256 = "25d361a8e0cdebd95279cf7058361e0de9fef6d4ce0c29b5113af42e91a20d0b"

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def read_parquet(path):
    import pyarrow.parquet as pq
    return pq.read_table(path,use_threads=False).to_pylist()

def schema_names(path):
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).schema_arrow.names

def write_parquet(path, rows):
    import pyarrow as pa, pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(list(rows)),path,compression="zstd")

def parse_date(x):
    if x is None:return None
    if isinstance(x,datetime):return x.date()
    if isinstance(x,date):return x
    s=str(x).strip()
    if not s or s.lower() in {"nan","none","nat","n/a"}:return None
    s=s.replace("年","-").replace("月","-").replace("日","").replace("/", "-").replace(".", "-")
    m=re.search(r"((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})",s)
    if not m:
        z=re.sub(r"\D","",s)
        if len(z)>=8:
            m=re.match(r"((?:19|20)\d{2})(\d{2})(\d{2})",z)
    if not m:return None
    try:return date(int(m.group(1)),int(m.group(2)),int(m.group(3)))
    except ValueError:return None

def parse_year(x):
    if x is None:return None
    try:
        if isinstance(x,(int,float)):
            y=int(x); return y if 1990<=y<=2030 else None
    except Exception:pass
    m=re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)",str(x))
    return int(m.group(1)) if m else None

def find_col(cols, candidates, contains=None):
    low={c.lower():c for c in cols}
    for x in candidates:
        if x.lower() in low:return low[x.lower()]
    if contains:
        for c in cols:
            lc=c.lower()
            if all(t.lower() in lc for t in contains):
                return c
    return None

def latest(parent,prefix,manifest):
    cand=[]
    for d in parent.glob(prefix+"*"):
        m=d/manifest
        if m.is_file():
            try:
                j=json.loads(m.read_text(encoding="utf-8"))
                if j.get("completed",True):
                    cand.append((m.stat().st_mtime,d,j))
            except Exception:pass
    if not cand:raise SystemExit(f"NO_COMPLETED:{parent}/{prefix}*")
    _,d,j=max(cand,key=lambda x:x[0])
    return d,j

def normalize_key(row, firm_col, year_col):
    f=row.get(firm_col)
    y=parse_year(row.get(year_col))
    if f is None or y is None:return None
    s=str(f).strip()
    m=re.search(r"(?<!\d)(\d{6})(?!\d)",s)
    if m:s=m.group(1)
    elif s.isdigit():s=f"{int(s):06d}"[-6:]
    return (s,y)

def reveal(path):
    if sys.platform=="darwin":
        subprocess.run(["open","-R",str(path)],
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage-d",action="store_true")
    ap.add_argument("--reveal",action="store_true")
    a=ap.parse_args()
    if not a.stage_d:ap.error("--stage-d required")

    os.umask(0o077)
    outroot=BASE/"label_v1_stageD";outroot.mkdir(parents=True,exist_ok=True,mode=0o700)
    out=outroot/("stageD_"+datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8]);out.mkdir(mode=0o700)
    report=out/"label_v1_stageD_report.txt"

    try:
        print("[1/6] 核验 legacy 标签文件和最新 Stage C/B3d……",flush=True)
        if sha256(LABEL_V08)!=V08_SHA256:raise ValueError("V08_HASH_MISMATCH")
        if sha256(LABEL_V0)!=V0_SHA256:raise ValueError("V0_HASH_MISMATCH")
        stageC,maniC=latest(BASE/"label_v1_stageC","stageC_","manifest_v1_stageC.json")
        stageB,maniB=latest(BASE/"label_v1_stageB3d","stageB3d_","manifest_v1_stageB3d.json")

        v08_cols=schema_names(LABEL_V08); v0_cols=schema_names(LABEL_V0)
        v08=read_parquet(LABEL_V08); v0=read_parquet(LABEL_V0)
        crows=read_parquet(stageC/"firm_year_evidence_v1_stageC.parquet")
        anchors=read_parquet(stageB/"annual_report_public_anchors_CNINFO_v4.parquet")

        print("[2/6] 自动识别 legacy schema 并精确分解 v07 / strict-added / loose-added……",flush=True)
        f08=find_col(v08_cols,["firm_id","company_code","stock_code","ts_code","symbol"])
        y08=find_col(v08_cols,["year","fiscal_year","report_year"])
        c_v07=find_col(v08_cols,["fraud_v07"])
        c_strict=find_col(v08_cols,["fraud_v08_strict"])
        c_loose=find_col(v08_cols,["fraud_v08_loose"])
        if not all([f08,y08,c_v07,c_strict,c_loose]):
            raise ValueError("V08_SCHEMA_UNRECOGNIZED:"+json.dumps(v08_cols,ensure_ascii=False))

        idx08={}
        for r in v08:
            k=normalize_key(r,f08,y08)
            if k:
                idx08[k]={
                    "v07":int(r.get(c_v07) or 0),
                    "strict":int(r.get(c_strict) or 0),
                    "loose":int(r.get(c_loose) or 0),
                }

        set_v07={k for k,v in idx08.items() if v["v07"]==1}
        set_strict={k for k,v in idx08.items() if v["strict"]==1}
        set_loose={k for k,v in idx08.items() if v["loose"]==1}
        strict_added=set_strict-set_v07
        loose_added=set_loose-set_strict

        print("[3/6] 恢复 v0 notice dates，检验“label year = notice year”及 post-filing 时序……",flush=True)
        f0=find_col(v0_cols,["firm_id","company_code","stock_code","ts_code","symbol"])
        y0=find_col(v0_cols,["year","fiscal_year","report_year"])
        first_col=find_col(v0_cols,["first_notice_date","first_announcement_date","first_date"],contains=["first","date"])
        last_col=find_col(v0_cols,["last_notice_date","last_announcement_date","last_date"],contains=["last","date"])
        med_col=find_col(v0_cols,["fraud_label_medium","fraud_medium","label_medium"])
        strict0_col=find_col(v0_cols,["fraud_label_strict","fraud_strict","label_strict"])
        loose0_col=find_col(v0_cols,["fraud_label_loose","fraud_loose","label_loose"])
        if not all([f0,y0,first_col,last_col]):
            raise ValueError("V0_SCHEMA_UNRECOGNIZED:"+json.dumps(v0_cols,ensure_ascii=False))

        idx0={}
        for r in v0:
            k=normalize_key(r,f0,y0)
            if not k:continue
            idx0[k]={
                "first":parse_date(r.get(first_col)),
                "last":parse_date(r.get(last_col)),
                "medium":int(r.get(med_col) or 0) if med_col else None,
                "strict0":int(r.get(strict0_col) or 0) if strict0_col else None,
                "loose0":int(r.get(loose0_col) or 0) if loose0_col else None,
            }

        amap={(r["company_code"],int(r["fiscal_year"])):parse_date(r.get("annual_report_public_anchor")) for r in anchors}

        v07_timing=Counter()
        v07_notice_year=Counter()
        v07_rows=[]
        for k in sorted(set_v07):
            o=idx0.get(k)
            anchor=amap.get(k)
            if o is None:
                v07_timing["no_v0_row"]+=1
                continue
            first=o["first"]; last=o["last"]
            if first:
                v07_notice_year["first_notice_calendar_year_equals_label_year" if first.year==k[1] else "first_notice_calendar_year_differs"]+=1
            else:
                v07_notice_year["first_notice_missing"]+=1

            if anchor is None:
                state="missing_CNINFO_anchor"
            elif first is None:
                state="missing_first_notice_date"
            elif first > anchor:
                state="first_notice_post_annual_report_anchor"
            elif first == anchor:
                state="first_notice_same_day_as_annual_report_anchor"
            else:
                state="first_notice_before_annual_report_anchor"
            v07_timing[state]+=1
            v07_rows.append({
                "company_code":k[0],
                "label_year":k[1],
                "first_notice_date":first.isoformat() if first else None,
                "last_notice_date":last.isoformat() if last else None,
                "annual_report_public_anchor":anchor.isoformat() if anchor else None,
                "timing_state":state,
            })

        write_parquet(out/"v07_notice_timing_STAGE_D_LOCAL_ONLY.parquet",v07_rows)

        print("[4/6] 对 strict-added 316 单独读取 Stage-C 证据状态，避免被 v07 淹没……",flush=True)
        cidx={}
        for r in crows:
            if r.get("scope")!="strict":continue
            k=(r["company_code"],int(r["fiscal_year"]))
            cidx[k]=r

        strict_added_states=Counter()
        v07_states=Counter()
        strict_all_states=Counter()
        for k in set_strict:
            st=(cidx.get(k) or {}).get("stageC_evidence_state","missing_stageC_row")
            strict_all_states[st]+=1
            if k in strict_added:strict_added_states[st]+=1
            if k in set_v07:v07_states[st]+=1

        print("[5/6] 计算 3-year maturity rule 的客观覆盖，不在此脚本最终冻结……",flush=True)
        maturity=Counter()
        maturity_year=defaultdict(Counter)
        for r in anchors:
            y=int(r["fiscal_year"])
            d=parse_date(r.get("annual_report_public_anchor"))
            if d is None:
                maturity["missing_anchor"]+=1
                maturity_year[y]["missing_anchor"]+=1
                continue
            days=(date(2026,5,8)-d).days
            for n,label in [(365,"ge_1y"),(730,"ge_2y"),(1095,"ge_3y"),(1460,"ge_4y")]:
                if days>=n:
                    maturity[label]+=1
                    maturity_year[y][label]+=1
            maturity["anchor_present"]+=1
            maturity_year[y]["anchor_present"]+=1

        print("[6/6] 写出关键报告；不生成 Label v1.0……",flush=True)
        lines=[
            "PeerJ 141707 | Label v1 Stage D legacy-source decomposition + temporal audit",
            "NO FINAL LABELS; NO TRAINING",
            "Local time: "+datetime.now().astimezone().isoformat(),
            "Stage C: "+str(stageC),
            "Stage B3d: "+str(stageB),
            "Output: "+str(out),"",
            "A. LEGACY LABEL SOURCE DECOMPOSITION",
            f"v07_positive_n={len(set_v07)}",
            f"strict_positive_n={len(set_strict)}",
            f"loose_positive_n={len(set_loose)}",
            f"v07_subset_of_strict={set_v07.issubset(set_strict)}",
            f"strict_subset_of_loose={set_strict.issubset(set_loose)}",
            f"strict_added_over_v07_n={len(strict_added)}",
            f"loose_added_over_strict_n={len(loose_added)}","",
            "B. V0 SCHEMA / NOTICE-DATE PROVENANCE",
            "v0_columns="+json.dumps(v0_cols,ensure_ascii=False),
            f"v0_first_notice_col={first_col}",
            f"v0_last_notice_col={last_col}",
            f"v0_medium_col={med_col}",
            "v07_notice_year_relation="+json.dumps(dict(v07_notice_year),ensure_ascii=False,sort_keys=True),
            "v07_notice_vs_CNINFO_anchor="+json.dumps(dict(v07_timing),ensure_ascii=False,sort_keys=True),"",
            "C. STAGE-C EVIDENCE STATE BY LEGACY SOURCE",
            "all_legacy_strict="+json.dumps(dict(strict_all_states),ensure_ascii=False,sort_keys=True),
            "v07_component="+json.dumps(dict(v07_states),ensure_ascii=False,sort_keys=True),
            "strict_added_over_v07="+json.dumps(dict(strict_added_states),ensure_ascii=False,sort_keys=True),"",
            "D. MATURITY DIAGNOSTIC (NOT YET FINAL RULE)",
            "overall="+json.dumps(dict(maturity),ensure_ascii=False,sort_keys=True),
            "by_year="+json.dumps({str(y):dict(v) for y,v in sorted(maturity_year.items())},ensure_ascii=False,sort_keys=True),"",
            "E. DECISION GATE",
            "Do NOT interpret Stage-C legacy1__no_machine_support as label errors before separating the v0.7 component.",
            "If v0.7 notice dates overwhelmingly equal the label calendar year and precede the CNINFO annual-report anchor, the historical v0.7 component is temporally incompatible with the current post-filing year-t target and cannot be unioned into Label v1.0 without remapping.",
            "Strict-added CSMAR cases are evaluated separately because they follow a different provenance path.",
            "Next: choose/remap the endpoint using these exact counts, then freeze maturity and generate Label v1.0.",
            "",
            "source_files_modified=false",
            "training_or_prediction_run=false",
            "final_nonnull_labels_written=0",
            "STATUS: LABEL_V1_STAGE_D_COMPLETE_READY_FOR_ENDPOINT_DECISION",
            "Upload only label_v1_stageD_report.txt.",
        ]
        report.write_text("\n".join(lines)+"\n",encoding="utf-8")

        (out/"manifest_v1_stageD.json").write_text(json.dumps({
            "completed":True,
            "created_at":datetime.now().astimezone().isoformat(),
            "v08_sha256":V08_SHA256,"v0_sha256":V0_SHA256,
            "stageC_manifest_sha256":sha256(stageC/"manifest_v1_stageC.json"),
            "stageB3d_manifest_sha256":sha256(stageB/"manifest_v1_stageB3d.json"),
            "final_labels_authorized":False,"training_authorized":False,
        },ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")

        print("STATUS: LABEL_V1_STAGE_D_COMPLETE_READY_FOR_ENDPOINT_DECISION")
        print("请只上传：",report)
        if a.reveal:reveal(report)

    except Exception as e:
        report.write_text(
            "PeerJ 141707 | Stage D FAILED SAFELY\n"
            f"ERROR_TYPE={type(e).__name__}\nERROR={e}\n"
            "source_files_modified=false\ntraining_or_prediction_run=false\nfinal_nonnull_labels_written=0\n"
            "STATUS: LABEL_V1_STAGE_D_FAILED_SAFE\n",encoding="utf-8")
        if a.reveal:reveal(report)
        raise

if __name__=="__main__":
    main()
