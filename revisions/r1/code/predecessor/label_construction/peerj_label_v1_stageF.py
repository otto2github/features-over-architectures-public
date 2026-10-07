#!/usr/bin/env python3
"""
PeerJ 141707 — Label v1.0 FINAL FREEZE (Stage F)

Primary endpoint policy
-----------------------
STRICT A+B:
  - strict violation types: P2501/P2502/P2503/P2507
  - formal CSRC HQ/branch administrative-penalty event
  - listed company supported as punished subject
  - fiscal year:
      A = periodic-report text supports structured ViolationYear
      B = CSMAR structured ViolationYear with no detected periodic-report-year conflict
  - DeclareDate strictly after CNINFO full annual-report public anchor
  - no pre/same-anchor DisposalDate conflict
  - historical v0.7 announcement-year source EXCLUDED

Primary risk-set maturity
-------------------------
>= 1095 days from CNINFO annual-report public anchor to fixed cutoff 2026-05-08.
Primary temporal split preserves the original train/validation boundaries:
  train      2010–2018
  validation 2019–2020
  test       2021–2022
2023–2024 remain nullable/right-censored for the primary endpoint.

Sensitivity columns
-------------------
- strict A-only: same endpoint but requires text-supported fiscal year
- loose A+B: loose type-code scope under the same timing/maturity rules

This script WRITES the frozen label artifact but DOES NOT run any model training.
It never overwrites source/project data. All outputs stay under:
  ~/peerj_141707_audit/label_v1_stageF/

On macOS --reveal opens Finder and selects the final report.
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
import subprocess
import sys
import uuid

BASE = Path.home() / "peerj_141707_audit"
VERSION = "label_v1.0"
CUTOFF = "2026-05-08"
MATURITY_DAYS = 1095

EXPECTED = {
    "strict_ab": {
        "train_2010_2018": {"n0": 23852, "n1": 254},
        "validation_2019_2020": {"n0": 7269, "n1": 40},
        "test_2021_2022": {"n0": 8415, "n1": 20},
    },
    "strict_aonly": {
        "train_2010_2018": {"n0": 23852, "n1": 218},
        "validation_2019_2020": {"n0": 7269, "n1": 36},
        "test_2021_2022": {"n0": 8415, "n1": 19},
    },
}

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

def latest(parent,prefix,manifest):
    cand=[]
    for d in parent.glob(prefix+"*"):
        m=d/manifest
        if not m.is_file(): continue
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

def split_for_year(y):
    if 2010<=y<=2018:return "train_2010_2018"
    if 2019<=y<=2020:return "validation_2019_2020"
    if 2021<=y<=2022:return "test_2021_2022"
    if 2023<=y<=2024:return "right_censored_2023_2024"
    return "outside_2010_2024"

def freeze_label(rec):
    """Primary/sensitivity label is only non-null where Stage-E candidate
    is eligible under >=1095-day maturity and the corresponding policy."""
    if not rec:
        return None
    if not bool(rec.get("primary_eligible_candidate")):
        return None
    x=rec.get("candidate_label")
    return int(x) if x in (0,1) else None

def reveal(path):
    if sys.platform=="darwin":
        subprocess.run(["open","-R",str(path)],
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage-f",action="store_true")
    ap.add_argument("--reveal",action="store_true")
    a=ap.parse_args()
    if not a.stage_f:ap.error("--stage-f required")

    os.umask(0o077)
    outroot=BASE/"label_v1_stageF"
    outroot.mkdir(parents=True,exist_ok=True,mode=0o700)
    out=outroot/("stageF_"+datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8])
    out.mkdir(mode=0o700)
    report=out/"label_v1_stageF_report.txt"

    try:
        print("[1/6] 读取并锁定最新 Stage E 候选风险集……",flush=True)
        stageE,maniE=latest(BASE/"label_v1_stageE","stageE_","manifest_v1_stageE.json")
        riskfile=stageE/"candidate_risksets_v1_stageE.parquet"
        if not riskfile.is_file():
            raise ValueError("STAGE_E_RISKSET_MISSING")
        erows=read_parquet(riskfile)

        # One record per firm-year/scope/policy.
        idx={}
        for r in erows:
            k=(r["company_code"],int(r["fiscal_year"]),r["scope"],r["policy"])
            if k in idx:
                raise ValueError(f"DUPLICATE_STAGE_E_KEY:{k}")
            idx[k]=r

        firmyears=sorted({(r["company_code"],int(r["fiscal_year"])) for r in erows},
                         key=lambda z:(z[1],z[0]))
        if len(firmyears)!=51675:
            raise ValueError(f"FIRMYEAR_COUNT_UNEXPECTED:{len(firmyears)}")

        print("[2/6] 冻结 primary=strict A+B；A-only/loose 仅作敏感性……",flush=True)
        outrows=[]
        counts=defaultdict(Counter)
        year_counts=defaultdict(Counter)
        legacy_cross=Counter()
        exclusions=Counter()

        for firm,y in firmyears:
            sab=idx.get((firm,y,"strict","A+B"))
            sao=idx.get((firm,y,"strict","A-only"))
            lab=idx.get((firm,y,"loose","A+B"))
            if sab is None or sao is None or lab is None:
                raise ValueError(f"MISSING_POLICY_ROW:{firm}:{y}")

            primary=freeze_label(sab)
            aonly=freeze_label(sao)
            loose=freeze_label(lab)
            split=split_for_year(y)

            # Invariant: no primary labels in immature 2023–2024.
            if y>=2023 and primary is not None:
                raise ValueError(f"IMMATURE_PRIMARY_LABEL_FOUND:{firm}:{y}")

            # A-only positives must be a subset of A+B positives.
            if aonly==1 and primary!=1:
                raise ValueError(f"AONLY_POS_NOT_SUBSET_AB:{firm}:{y}")

            # Freeze status explains nulls without converting them to negatives.
            if primary is not None:
                freeze_status="primary_eligible_labeled"
            elif y>=2023:
                freeze_status="right_censored_primary_null"
            elif not sab.get("annual_report_public_anchor"):
                freeze_status="missing_public_anchor_primary_null"
            elif str(sab.get("evidence_state") or "").startswith("unresolved_"):
                freeze_status="unresolved_evidence_primary_null"
            elif not sab.get("mature_ge_1095d"):
                freeze_status="insufficient_followup_primary_null"
            else:
                freeze_status="other_ineligible_primary_null"
            exclusions[freeze_status]+=1

            legacy=int(sab["legacy_label"])
            if primary is None:
                legacy_cross[f"legacy{legacy}__newNULL"]+=1
            else:
                legacy_cross[f"legacy{legacy}__new{primary}"]+=1

            if primary is not None:
                counts[split][f"primary_{primary}"]+=1
                counts[split]["primary_n"]+=1
                year_counts[y][f"primary_{primary}"]+=1
                year_counts[y]["primary_n"]+=1
            else:
                counts[split]["primary_null"]+=1
                year_counts[y]["primary_null"]+=1

            if aonly is not None:
                counts[split][f"aonly_{aonly}"]+=1
                counts[split]["aonly_n"]+=1
                year_counts[y][f"aonly_{aonly}"]+=1
            else:
                counts[split]["aonly_null"]+=1
                year_counts[y]["aonly_null"]+=1

            if loose is not None:
                counts[split][f"loose_{loose}"]+=1
                counts[split]["loose_n"]+=1
                year_counts[y][f"loose_{loose}"]+=1
            else:
                counts[split]["loose_null"]+=1
                year_counts[y]["loose_null"]+=1

            outrows.append({
                "company_code":firm,
                "fiscal_year":y,
                "split_v1":split,
                "annual_report_public_anchor":sab.get("annual_report_public_anchor"),
                "followup_days_to_2026_05_08":sab.get("followup_days_to_2026_05_08"),
                "mature_ge_1095d":bool(sab.get("mature_ge_1095d")),
                "label_v1_strict_ab_primary":primary,
                "eligible_v1_strict_ab_primary":primary is not None,
                "evidence_state_strict_ab":sab.get("evidence_state"),
                "label_v1_strict_aonly_sensitivity":aonly,
                "eligible_v1_strict_aonly_sensitivity":aonly is not None,
                "evidence_state_strict_aonly":sao.get("evidence_state"),
                "label_v1_loose_ab_sensitivity":loose,
                "eligible_v1_loose_ab_sensitivity":loose is not None,
                "evidence_state_loose_ab":lab.get("evidence_state"),
                "legacy_strict_label":legacy,
                "freeze_status":freeze_status,
                "cutoff_date":CUTOFF,
                "maturity_threshold_days":MATURITY_DAYS,
                "label_version":VERSION,
            })

        print("[3/6] 执行冻结前硬性计数核验……",flush=True)
        for split,exp in EXPECTED["strict_ab"].items():
            got0=counts[split]["primary_0"]; got1=counts[split]["primary_1"]
            if (got0,got1)!=(exp["n0"],exp["n1"]):
                raise ValueError(
                    f"STRICT_AB_COUNT_MISMATCH:{split}:got({got0},{got1})"
                    f":expected({exp['n0']},{exp['n1']})"
                )
        for split,exp in EXPECTED["strict_aonly"].items():
            got0=counts[split]["aonly_0"]; got1=counts[split]["aonly_1"]
            if (got0,got1)!=(exp["n0"],exp["n1"]):
                raise ValueError(
                    f"STRICT_AONLY_COUNT_MISMATCH:{split}:got({got0},{got1})"
                    f":expected({exp['n0']},{exp['n1']})"
                )

        primary_n=sum(c["primary_n"] for c in counts.values())
        primary_pos=sum(c["primary_1"] for c in counts.values())
        primary_neg=sum(c["primary_0"] for c in counts.values())
        # Only 2010–2022 can be primary-labeled.
        if primary_n!=39850 or primary_pos!=314 or primary_neg!=39536:
            raise ValueError(
                f"PRIMARY_TOTAL_MISMATCH:n={primary_n},pos={primary_pos},neg={primary_neg}"
            )

        print("[4/6] 写入冻结标签 parquet 与审计摘要……",flush=True)
        labelfile=out/"fraud_labels_v1_0.parquet"
        write_parquet(labelfile,outrows)
        label_sha=sha256(labelfile)

        year_csv=out/"label_v1_0_year_summary.csv"
        with year_csv.open("w",encoding="utf-8",newline="") as f:
            fields=[
                "fiscal_year","primary_n","primary_0","primary_1","primary_null",
                "aonly_n","aonly_0","aonly_1","aonly_null",
                "loose_n","loose_0","loose_1","loose_null",
            ]
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for y in range(2010,2025):
                c=year_counts[y]
                w.writerow({"fiscal_year":y,**{k:c[k] for k in fields if k!="fiscal_year"}})

        print("[5/6] 生成 legacy diff；明确它不是错误率估计……",flush=True)
        legacy_panel_pos=sum(1 for r in outrows if r["legacy_strict_label"]==1)
        new_pos_keys={(r["company_code"],r["fiscal_year"]) for r in outrows if r["label_v1_strict_ab_primary"]==1}
        old_pos_keys={(r["company_code"],r["fiscal_year"]) for r in outrows if r["legacy_strict_label"]==1}
        overlap=len(new_pos_keys & old_pos_keys)
        new_only=len(new_pos_keys-old_pos_keys)
        legacy_only=len(old_pos_keys-new_pos_keys)

        print("[6/6] 写出最终 freeze report 与 manifest……",flush=True)
        split_json={k:dict(v) for k,v in counts.items()}
        year_json={str(k):dict(v) for k,v in sorted(year_counts.items())}
        lines=[
            "PeerJ 141707 | Label v1.0 FINAL FREEZE (Stage F)",
            "LABEL ARTIFACT WRITTEN; NO MODEL TRAINING",
            "Local time: "+datetime.now().astimezone().isoformat(),
            "Stage E: "+str(stageE),
            "Output: "+str(out),"",
            "A. PRIMARY POLICY FREEZE",
            "label_version=label_v1.0",
            "primary_scope=strict",
            "primary_year_policy=A+B",
            "primary_year_policy_reason=CSMAR ViolationYear is source-defined as actual violation year; B is accepted only when no periodic-report-year conflict is detected. A-only is retained as sensitivity.",
            "historical_v07_union=false",
            "mechanical_v07_year_shift=false",
            "cutoff_date=2026-05-08",
            "maturity_threshold_days=1095",
            "primary_train_years=2010-2018",
            "primary_validation_years=2019-2020",
            "primary_test_years=2021-2022",
            "right_censored_years=2023-2024",
            "negative_definition=no qualifying formal-CSRC endpoint observed by cutoff among mature rows with no unresolved potential final event","",
            "B. FROZEN PRIMARY COUNTS",
            f"primary_labeled_n={primary_n}",
            f"primary_positive_n={primary_pos}",
            f"primary_negative_n={primary_neg}",
            f"primary_prevalence={primary_pos/primary_n:.8f}",
            "by_split="+json.dumps(split_json,ensure_ascii=False,sort_keys=True),
            "by_year="+json.dumps(year_json,ensure_ascii=False,sort_keys=True),"",
            "C. SENSITIVITY POLICY",
            "strict_A_only=retained as year-mapping sensitivity",
            "loose_A_plus_B=retained as type-scope sensitivity","",
            "D. LEGACY DIFF (DESCRIPTIVE, NOT ERROR ADJUDICATION)",
            f"legacy_strict_panel_positive_n={legacy_panel_pos}",
            f"new_primary_positive_n={len(new_pos_keys)}",
            f"positive_key_overlap={overlap}",
            f"new_primary_positive_not_legacy_positive={new_only}",
            f"legacy_positive_not_new_primary_positive_or_outside_new_riskset={legacy_only}",
            "legacy_cross="+json.dumps(dict(legacy_cross),ensure_ascii=False,sort_keys=True),
            "Do not interpret legacy-only keys as false labels; most legacy positives came from the historically incompatible v0.7 announcement-year source.","",
            "E. NULL / EXCLUSION STATUS",
            json.dumps(dict(exclusions),ensure_ascii=False,sort_keys=True),"",
            "F. ARTIFACT",
            "label_file="+str(labelfile),
            "label_sha256="+label_sha,
            f"label_rows={len(outrows)}",
            "source_files_modified=false",
            "model_training_run=false",
            "STATUS: LABEL_V1_0_FROZEN_READY_FOR_FORMAL_RERUN",
            "Upload only label_v1_stageF_report.txt.",
        ]
        report.write_text("\n".join(lines)+"\n",encoding="utf-8")

        manifest={
            "completed":True,
            "frozen":True,
            "created_at":datetime.now().astimezone().isoformat(),
            "label_version":VERSION,
            "primary_policy":"strict_A+B",
            "strict_A_only_sensitivity":True,
            "loose_A+B_sensitivity":True,
            "historical_v07_union":False,
            "maturity_days":MATURITY_DAYS,
            "cutoff_date":CUTOFF,
            "train_years":[2010,2018],
            "validation_years":[2019,2020],
            "test_years":[2021,2022],
            "right_censored_years":[2023,2024],
            "stageE_manifest_sha256":sha256(stageE/"manifest_v1_stageE.json"),
            "riskset_input_sha256":sha256(riskfile),
            "label_file":"fraud_labels_v1_0.parquet",
            "label_sha256":label_sha,
            "label_rows":len(outrows),
            "primary_labeled_n":primary_n,
            "primary_positive_n":primary_pos,
            "primary_negative_n":primary_neg,
            "training_authorized_after_independent_artifact_check":False,
        }
        (out/"manifest_v1_stageF.json").write_text(
            json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n",
            encoding="utf-8"
        )

        print("STATUS: LABEL_V1_0_FROZEN_READY_FOR_FORMAL_RERUN")
        print("请只上传：",report)
        if a.reveal:reveal(report)

    except Exception as e:
        report.write_text(
            "PeerJ 141707 | Stage F FAILED SAFELY\n"
            f"ERROR_TYPE={type(e).__name__}\nERROR={e}\n"
            "source_files_modified=false\nmodel_training_run=false\n"
            "STATUS: LABEL_V1_STAGE_F_FAILED_SAFE\n",
            encoding="utf-8"
        )
        if a.reveal:reveal(report)
        raise

if __name__=="__main__":
    main()
