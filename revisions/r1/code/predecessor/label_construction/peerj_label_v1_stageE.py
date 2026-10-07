#!/usr/bin/env python3
"""
PeerJ 141707 — Label v1 Stage E
Endpoint qualification rebuild after Stage-D legacy-source audit.

Purpose
-------
Stage D established that historical v0.7 is an announcement-year label source,
not a fiscal-year-t post-filing label source. Stage E therefore does NOT union
v0.7 into the new endpoint. Instead it rebuilds the candidate endpoint from the
traceable CSMAR event source, using the frozen CNINFO annual-report public anchor.

This is still a CANDIDATE/FREEZE-AUDIT stage:
- no final Label v1.0 is written;
- no model training is run;
- no source file is modified.

Two candidate evidence policies are reported:
  A-only : explicit periodic-report-year text supports the structured ViolationYear.
  A+B    : A plus structured ViolationYear when no periodic-report-year conflict is
           detected. This is descriptor-supported but weaker than A.

Both policies require:
- strict/loose violation type scope;
- final administrative-penalty document stage;
- CSRC HQ/branch authority, including a conservative text rescue for company
  announcements explicitly naming a CSRC/证监局 administrative-penalty decision;
- listed-company subject evidence;
- DeclareDate strictly after the CNINFO annual-report public anchor;
- no pre/same-anchor DisposalDate conflict.

Primary maturity diagnostic: >= 1095 days follow-up to 2026-05-08.
2023–2024 are not made mature merely because a positive is observed.

Writes only under ~/peerj_141707_audit/label_v1_stageE/
On macOS, --reveal selects the final report in Finder.
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
CUTOFF = date(2026, 5, 8)
MATURE_DAYS = 1095

QUAL_SUBJECT = {
    "confirmed_fullname_in_final_measure",
    "confirmed_shortname_in_final_measure",
    "supported_by_company_punishment_field",
    "probable_you_company_final_measure",
}
PRE_A = "A_text_year_supported_pre_anchor"
PRE_B = "B_raw_year_only_pre_anchor"

def sha256(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def read_parquet(path):
    import pyarrow.parquet as pq
    return pq.read_table(path,use_threads=False).to_pylist()

def write_parquet(path,rows):
    import pyarrow as pa, pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(list(rows)),path,compression="zstd")

def read_jsonl(path):
    with path.open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

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

def latest(parent,prefix,manifest):
    cand=[]
    for d in parent.glob(prefix+"*"):
        m=d/manifest
        if not m.is_file():continue
        try:
            j=json.loads(m.read_text(encoding="utf-8"))
            if j.get("completed",True):
                cand.append((m.stat().st_mtime,d,j))
        except Exception:pass
    if not cand:
        raise SystemExit(f"NO_COMPLETED_RUN:{parent}/{prefix}*")
    _,d,j=max(cand,key=lambda x:x[0])
    return d,j

def authority_text_rescue(raw):
    """Conservative rescue only when a final-penalty phrase is explicitly tied
    to CSRC/证监局 wording. Generic '收到行政处罚决定书' is NOT enough."""
    title=str(raw.get("FileName") or "")
    supervisor=str(raw.get("Supervisor") or "")
    promulgator=str(raw.get("Promulgator") or "")
    activity=str(raw.get("Activity") or "")
    measure=str(raw.get("PunishmentMeasure") or "")
    # Company-announcement titles and opening factual text are the highest-value rescue locations.
    text="\n".join([title,supervisor,promulgator,activity[:1600],measure[:800]])
    agency=r"(?:中国证券监督管理委员会(?:[^。\n]{0,24}监管局)?|中国证监会|[\u4e00-\u9fff]{2,10}证监局)"
    final=r"(?:行政处罚决定书|行政处罚决定)"
    if re.search(agency+r"[^。\n]{0,80}"+final,text):
        return True
    if re.search(final+r"[^。\n]{0,80}"+agency,text):
        return True
    # Common title: 关于收到XX证监局《行政处罚决定书》的公告
    if re.search(r"收到[^。\n]{0,80}"+agency+r"[^。\n]{0,80}"+final,title):
        return True
    return False

def classify_pair(p, raw, anchor):
    """Return candidate endpoint status for one type-year pair."""
    # Outside-type/unresolved-year stay outside the endpoint build.
    pre=p.get("preanchor_status")
    if pre=="outside_type_scope":
        return "outside_type_scope", None
    if pre=="unresolved_year":
        return "unresolved_year", None

    # Final administrative penalty only.
    stage=p.get("document_stage")
    if not str(stage or "").startswith("final_penalty"):
        return "nonfinal_or_nonendpoint_document", None

    # Authority: structured first; conservative text rescue second.
    auth=p.get("authority_class")
    if auth in {"csrc_hq","csrc_branch"}:
        auth_grade="structured_csrc"
    elif authority_text_rescue(raw):
        auth_grade="text_rescued_csrc_final_decision"
    else:
        return "unresolved_final_penalty_authority", None

    # Listed company itself must be a supported subject.
    if p.get("company_subject_status") not in QUAL_SUBJECT:
        return "unresolved_company_subject", auth_grade

    # Year evidence: conflict blocks automatic qualification; raw-only is grade B.
    ys=p.get("year_evidence_status")
    if ys=="raw_year_conflicts_with_periodic_report_text":
        return "unresolved_year_conflict", auth_grade
    if ys=="raw_year_supported_by_periodic_report_text":
        grade="A"
    elif ys in {"raw_year_only_no_periodic_report_year_extracted"}:
        grade="B"
    else:
        return "unresolved_year_evidence", auth_grade

    if anchor is None:
        return "unresolved_missing_annual_report_anchor", (grade,auth_grade)

    decl=parse_date(p.get("declare_date"))
    disp=parse_date(p.get("disposal_date"))
    if decl is None:
        return "unresolved_missing_public_sanction_date", (grade,auth_grade)
    if decl <= anchor:
        return "public_sanction_on_or_before_anchor", (grade,auth_grade)
    if disp is not None and disp <= anchor:
        return "timing_conflict_document_date_pre_or_same", (grade,auth_grade)

    if grade=="A":
        return "QUALIFIED_A_POSTFILING_FORMAL_CSRC", (grade,auth_grade)
    return "QUALIFIED_B_POSTFILING_FORMAL_CSRC", (grade,auth_grade)

def policy_state(pair_states, policy):
    s=set(pair_states)
    if "QUALIFIED_A_POSTFILING_FORMAL_CSRC" in s:
        return "positive_A"
    if "QUALIFIED_B_POSTFILING_FORMAL_CSRC" in s:
        return "positive_B" if policy=="A+B" else "unresolved_B_not_accepted"
    blockers={
        "unresolved_year",
        "unresolved_final_penalty_authority",
        "unresolved_company_subject",
        "unresolved_year_conflict",
        "unresolved_year_evidence",
        "unresolved_missing_annual_report_anchor",
        "unresolved_missing_public_sanction_date",
        "timing_conflict_document_date_pre_or_same",
    }
    hit=sorted(s & blockers)
    if hit:
        return "unresolved_potential_final_event:"+"|".join(hit)
    # pre-filing final penalty, non-final measures/notices, exchange measures, and
    # out-of-scope codes do not themselves make a positive under this endpoint.
    return "no_qualifying_endpoint_observed"

def reveal(path):
    if sys.platform=="darwin":
        subprocess.run(["open","-R",str(path)],
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage-e",action="store_true")
    ap.add_argument("--reveal",action="store_true")
    a=ap.parse_args()
    if not a.stage_e:ap.error("--stage-e required")

    os.umask(0o077)
    outroot=BASE/"label_v1_stageE";outroot.mkdir(parents=True,exist_ok=True,mode=0o700)
    out=outroot/("stageE_"+datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8])
    out.mkdir(mode=0o700)
    report=out/"label_v1_stageE_report.txt"

    try:
        print("[1/7] 读取 Stage A / B3d / C / D，并锁定同一来源链……",flush=True)
        stageA,maniA=latest(BASE/"label_v1_stageA","stageA_","manifest_v1_stageA.json")
        stageB,maniB=latest(BASE/"label_v1_stageB3d","stageB3d_","manifest_v1_stageB3d.json")
        stageC,maniC=latest(BASE/"label_v1_stageC","stageC_","manifest_v1_stageC.json")
        stageD,maniD=latest(BASE/"label_v1_stageD","stageD_","manifest_v1_stageD.json")

        protocol=json.loads((stageA/"protocol_v1_stageA.json").read_text(encoding="utf-8"))
        pkg=Path(protocol["source_package"])
        if not pkg.is_dir():
            raise ValueError("STAGE_A_SOURCE_PACKAGE_MISSING")
        if sha256(pkg/"manifest.json") != maniA["candidate_package_manifest_sha256"]:
            raise ValueError("CANDIDATE_PACKAGE_MANIFEST_HASH_MISMATCH")

        anchor_file=stageB/"annual_report_public_anchors_CNINFO_v4.parquet"
        if sha256(anchor_file) != maniB["anchor_sha256"]:
            raise ValueError("ANCHOR_HASH_MISMATCH")

        print("[2/7] 加载原始事件文本，仅用于保守的CSRC机关文本救援……",flush=True)
        raw_by_ref={}
        for ev in read_jsonl(pkg/"event_evidence.jsonl"):
            raw_by_ref[ev["row_ref"]]=ev.get("raw",{})
        if not raw_by_ref:
            raise ValueError("EMPTY_EVENT_EVIDENCE")

        anchors=read_parquet(anchor_file)
        amap={(r["company_code"],int(r["fiscal_year"])):parse_date(r.get("annual_report_public_anchor")) for r in anchors}

        crows=read_parquet(stageC/"firm_year_evidence_v1_stageC.parquet")
        panel_meta={}
        for r in crows:
            key=(r["scope"],r["company_code"],int(r["fiscal_year"]))
            panel_meta[key]={
                "partition":r["partition"],
                "legacy_label":int(r["legacy_label"]),
            }

        print("[3/7] 对 Stage-A pair ledger 重新执行正式处罚/机关/主体/年度/post-filing资格……",flush=True)
        pairs=read_parquet(stageA/"pair_ledger_v1_stageA.parquet")
        pair_out=[]
        status_counts={"strict":Counter(),"loose":Counter()}
        rescue_counts={"strict":Counter(),"loose":Counter()}
        states_by_key=defaultdict(list)

        for p in pairs:
            scope=p["scope"]
            yr=p.get("raw_violation_year")
            try: yr=int(yr) if yr is not None else None
            except Exception: yr=None
            key=(p.get("company_code"),yr) if p.get("company_code") and yr is not None else None
            anchor=amap.get(key) if key else None
            raw=raw_by_ref.get(p["row_ref"],{})
            st,detail=classify_pair(p,raw,anchor)
            status_counts[scope][st]+=1
            if detail and isinstance(detail,tuple) and len(detail)==2:
                grade,auth_grade=detail
                if auth_grade=="text_rescued_csrc_final_decision":
                    rescue_counts[scope][st]+=1
            if key and 2010<=yr<=2024:
                states_by_key[(scope,key)].append(st)

            q=dict(p)
            q["annual_report_public_anchor"]=anchor.isoformat() if anchor else None
            q["stageE_pair_status"]=st
            q["final_label"]=None
            q["training_ready"]=False
            pair_out.append(q)

        write_parquet(out/"pair_qualification_v1_stageE.parquet",pair_out)

        print("[4/7] 形成 A-only 与 A+B 两套候选风险集；不冻结最终标签……",flush=True)
        policy_rows=[]
        policy_counts={"A-only":Counter(),"A+B":Counter()}
        by_partition={"A-only":defaultdict(Counter),"A+B":defaultdict(Counter)}
        legacy_cross={"A-only":Counter(),"A+B":Counter()}
        unresolved_private=[]

        for scope in ("strict","loose"):
            keys=[(f,y) for s,f,y in panel_meta if s==scope]
            for firm,yr in keys:
                meta=panel_meta[(scope,firm,yr)]
                anchor=amap.get((firm,yr))
                follow=(CUTOFF-anchor).days if anchor else None
                mature=anchor is not None and follow>=MATURE_DAYS
                pair_states=states_by_key.get((scope,(firm,yr)),[])
                for pol in ("A-only","A+B"):
                    st=policy_state(pair_states,pol)
                    positive=st in {"positive_A","positive_B"}
                    unresolved=st.startswith("unresolved_")
                    if positive:
                        candidate_label=1
                    elif mature and not unresolved:
                        candidate_label=0
                    else:
                        candidate_label=None

                    # Primary cohort is mature rows through FY2022 only.
                    # 2023–2024 remain outside primary evaluation even if a positive is known.
                    primary_eligible = bool(
                        candidate_label is not None and mature and yr<=2022
                    )

                    rec={
                        "company_code":firm,
                        "fiscal_year":yr,
                        "scope":scope,
                        "partition_original":meta["partition"],
                        "policy":pol,
                        "evidence_state":st,
                        "annual_report_public_anchor":anchor.isoformat() if anchor else None,
                        "followup_days_to_2026_05_08":follow,
                        "mature_ge_1095d":mature,
                        "candidate_label":candidate_label,
                        "primary_eligible_candidate":primary_eligible,
                        "legacy_label":meta["legacy_label"],
                        "final_label":None,
                        "training_ready":False,
                    }
                    policy_rows.append(rec)
                    k=f"{scope}__{st}"
                    policy_counts[pol][k]+=1
                    by_partition[pol][meta["partition"]][f"{scope}__{st}"]+=1
                    legacy_cross[pol][f"{scope}__legacy{meta['legacy_label']}__{st}"]+=1
                    if unresolved and scope=="strict":
                        unresolved_private.append({
                            "company_code":"'"+firm,"fiscal_year":yr,
                            "policy":pol,"partition":meta["partition"],
                            "legacy_label":meta["legacy_label"],
                            "evidence_state":st,
                            "anchor":rec["annual_report_public_anchor"],
                            "followup_days":follow,
                        })

        write_parquet(out/"candidate_risksets_v1_stageE.parquet",policy_rows)

        if unresolved_private:
            with (out/"strict_unresolved_STAGEE_LOCAL_ONLY.csv").open("w",encoding="utf-8",newline="") as f:
                fields=list(unresolved_private[0])
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(unresolved_private)

        print("[5/7] 计算真正可用于训练/验证/成熟测试的候选样本规模……",flush=True)
        cohort={"A-only":defaultdict(Counter),"A+B":defaultdict(Counter)}
        for r in policy_rows:
            if r["scope"]!="strict":continue
            pol=r["policy"]
            y=r["fiscal_year"]
            if y<=2018:split="train_2010_2018"
            elif y<=2020:split="validation_2019_2020"
            elif y<=2022:split="mature_test_2021_2022"
            else:split="immature_2023_2024"
            if r["candidate_label"] is None:
                cohort[pol][split]["unresolved_or_ineligible"]+=1
            else:
                cohort[pol][split]["labeled_candidate"]+=1
                cohort[pol][split][f"label_{r['candidate_label']}"]+=1
                if r["primary_eligible_candidate"]:
                    cohort[pol][split]["primary_eligible"]+=1
                    cohort[pol][split][f"primary_label_{r['candidate_label']}"]+=1

        print("[6/7] 固化 Stage-D endpoint结论：v0.7不并入新主标签……",flush=True)
        # Stage-D report is authoritative for these counts; parse exact lines.
        dtext=(stageD/"label_v1_stageD_report.txt").read_text(encoding="utf-8")
        def grab(name):
            m=re.search(rf"^{re.escape(name)}=(.+)$",dtext,re.M)
            return m.group(1).strip() if m else "NA"
        v07_n=grab("v07_positive_n")
        v07_relation=grab("v07_notice_year_relation")
        v07_timing=grab("v07_notice_vs_CNINFO_anchor")

        print("[7/7] 写出决策报告；仍不授权正式训练……",flush=True)
        lines=[
            "PeerJ 141707 | Label v1 Stage E endpoint qualification rebuild",
            "NO FINAL LABELS; NO TRAINING",
            "Local time: "+datetime.now().astimezone().isoformat(),
            "Stage A: "+str(stageA),
            "Stage B3d: "+str(stageB),
            "Stage C: "+str(stageC),
            "Stage D: "+str(stageD),
            "Output: "+str(out),"",
            "A. ENDPOINT DECISION FROM STAGE D",
            "historical_v07_in_primary_candidate=false",
            "reason=v0.7 is announcement-year provenance and is temporally incompatible with the current fiscal-year-t post-filing endpoint unless independently remapped from source documents.",
            f"v07_positive_n={v07_n}",
            "v07_notice_year_relation="+v07_relation,
            "v07_notice_vs_CNINFO_anchor="+v07_timing,
            "mechanical_year_minus_one_remap_allowed=false","",
            "B. REBUILT PAIR QUALIFICATION COUNTS",
            "strict="+json.dumps(dict(status_counts["strict"]),ensure_ascii=False,sort_keys=True),
            "loose="+json.dumps(dict(status_counts["loose"]),ensure_ascii=False,sort_keys=True),
            "strict_text_authority_rescues="+json.dumps(dict(rescue_counts["strict"]),ensure_ascii=False,sort_keys=True),
            "loose_text_authority_rescues="+json.dumps(dict(rescue_counts["loose"]),ensure_ascii=False,sort_keys=True),"",
            "C. FIRM-YEAR POLICY COUNTS",
            "A-only="+json.dumps(dict(policy_counts["A-only"]),ensure_ascii=False,sort_keys=True),
            "A+B="+json.dumps(dict(policy_counts["A+B"]),ensure_ascii=False,sort_keys=True),"",
            "D. STRICT CANDIDATE COHORTS",
            "A-only="+json.dumps({k:dict(v) for k,v in cohort["A-only"].items()},ensure_ascii=False,sort_keys=True),
            "A+B="+json.dumps({k:dict(v) for k,v in cohort["A+B"].items()},ensure_ascii=False,sort_keys=True),"",
            "E. MATURITY RULE PROPOSAL",
            f"candidate_maturity_threshold_days={MATURE_DAYS}",
            "primary_train_years=2010-2018",
            "primary_validation_years=2019-2020",
            "primary_mature_test_years=2021-2022",
            "2023-2024_role=right-censored temporal sensitivity/descriptive only; not primary mature test",
            "negative_definition_candidate=no qualifying formal-CSRC endpoint observed by cutoff among mature rows, provided no unresolved potential final event remains",
            "",
            "F. FREEZE GATE",
            "Stage E does not choose between A-only and A+B automatically.",
            "A-only is the more conservative text-supported fiscal-year mapping.",
            "A+B additionally trusts CSMAR structured ViolationYear when no periodic-report-year conflict is detected.",
            "Use the cohort counts above to choose the primary mapping before Label v1.0 is written.",
            "",
            "source_files_modified=false",
            "training_or_prediction_run=false",
            "final_nonnull_labels_written=0",
            "STATUS: LABEL_V1_STAGE_E_COMPLETE_READY_FOR_PRIMARY_POLICY_FREEZE",
            "Upload only label_v1_stageE_report.txt.",
        ]
        report.write_text("\n".join(lines)+"\n",encoding="utf-8")

        (out/"manifest_v1_stageE.json").write_text(json.dumps({
            "completed":True,
            "created_at":datetime.now().astimezone().isoformat(),
            "stageA_manifest_sha256":sha256(stageA/"manifest_v1_stageA.json"),
            "stageB3d_manifest_sha256":sha256(stageB/"manifest_v1_stageB3d.json"),
            "stageC_manifest_sha256":sha256(stageC/"manifest_v1_stageC.json"),
            "stageD_manifest_sha256":sha256(stageD/"manifest_v1_stageD.json"),
            "maturity_days_candidate":MATURE_DAYS,
            "historical_v07_in_primary_candidate":False,
            "final_labels_authorized":False,
            "training_authorized":False,
        },ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")

        print("STATUS: LABEL_V1_STAGE_E_COMPLETE_READY_FOR_PRIMARY_POLICY_FREEZE")
        print("请只上传：",report)
        if a.reveal:reveal(report)

    except Exception as e:
        report.write_text(
            "PeerJ 141707 | Stage E FAILED SAFELY\n"
            f"ERROR_TYPE={type(e).__name__}\nERROR={e}\n"
            "source_files_modified=false\ntraining_or_prediction_run=false\nfinal_nonnull_labels_written=0\n"
            "STATUS: LABEL_V1_STAGE_E_FAILED_SAFE\n",encoding="utf-8")
        if a.reveal:reveal(report)
        raise

if __name__=="__main__":
    main()
