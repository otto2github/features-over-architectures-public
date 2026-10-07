#!/usr/bin/env python3
"""
PeerJ 141707 — Label v1 Stage B1
Targeted adjudication of the best annual-report filing-anchor candidate.

Reads:
  - latest completed/stopped Stage B anchor_candidates_v1_stageB_LOCAL_ONLY.csv
  - the selected candidate XLSX
  - descriptor text files in the same CSMAR directory
  - pinned 51,675 firm-year panel

Writes only under:
  ~/peerj_141707_audit/label_v1_stageB1/

No training. No final labels. No source overwrite. No network.
On macOS, --reveal opens Finder and selects the final report.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, re, stat, subprocess, sys, time, uuid, zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

BASE = Path.home() / "peerj_141707_audit"
PANEL = Path(os.environ.get("PEERJ_SOURCE_PROJECT_ROOT", "private_inputs/project") + '/data/processed/kg/node_features_v1_1.parquet')
PANEL_SHA256 = "adc00129f06ffb0995691c6d1ba98f740c53561c1e0cc4e91d927b84eb36beb2"
EXPECTED_N = 51675
MAX_SECONDS = 900
START = time.monotonic()

def tick():
    if time.monotonic() - START > MAX_SECONDS:
        raise RuntimeError("TIME_LIMIT_900_SECONDS")

def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def parse_code(x):
    if x is None: return None
    s = str(x).strip()
    if re.fullmatch(r"\d+(?:\.0+)?", s):
        try: return f"{int(float(s)):06d}"[-6:]
        except Exception: pass
    m = re.search(r"(?<!\d)(\d{6})(?!\d)", s)
    return m.group(1) if m else None

def excel_date(x):
    try: v = float(x)
    except Exception: return None
    if 20000 <= v <= 80000:
        return date(1899,12,30) + timedelta(days=int(v))
    return None

def parse_date(x):
    if x is None: return None
    if isinstance(x, datetime): return x.date()
    if isinstance(x, date): return x
    d = excel_date(x)
    if d: return d
    s = str(x).strip().replace("年","-").replace("月","-").replace("日","")
    s = s.replace("/", "-").replace(".", "-")
    m = re.search(r"((?:19|20)\d{2})-?(\d{1,2})-?(\d{1,2})", s)
    if not m: return None
    try: return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError: return None

def parse_year(x):
    if x is None: return None
    try:
        if isinstance(x, (int,float)):
            y=int(x)
            return y if 1990<=y<=2030 else None
    except Exception: pass
    m = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", str(x))
    return int(m.group(1)) if m else None

def cell_col(ref):
    m=re.match(r"([A-Z]+)", ref or "")
    if not m: return None
    n=0
    for ch in m.group(1): n=n*26+ord(ch)-64
    return n-1

def shared_strings(z):
    ns="{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    out=[]
    try:
        root=ET.parse(z.open("xl/sharedStrings.xml")).getroot()
        for si in root.findall(ns+"si"):
            out.append("".join((t.text or "") for t in si.iter(ns+"t")))
    except KeyError:
        pass
    return out

def sheet_targets(z):
    nsm="{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    nsr="{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    nsp="{http://schemas.openxmlformats.org/package/2006/relationships}"
    wb=ET.parse(z.open("xl/workbook.xml")).getroot()
    rr=ET.parse(z.open("xl/_rels/workbook.xml.rels")).getroot()
    rel={r.attrib["Id"]:r.attrib["Target"] for r in rr.findall(nsp+"Relationship")}
    out=[]
    for sh in wb.find(nsm+"sheets"):
        target=rel.get(sh.attrib.get(nsr+"id"))
        if target:
            if not target.startswith("xl/"): target="xl/"+target.lstrip("/")
            out.append((sh.attrib.get("name",""),target))
    return out

def iter_xlsx_rows(path: Path, target_sheet: str):
    ns="{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as z:
        ss=shared_strings(z)
        target=None
        for name,p in sheet_targets(z):
            if name==target_sheet:
                target=p; break
        if target is None:
            raise ValueError(f"SHEET_NOT_FOUND:{target_sheet}")
        with z.open(target) as f:
            for ev,elem in ET.iterparse(f,events=("end",)):
                if elem.tag!=ns+"row": continue
                vals={}
                for c in elem.findall(ns+"c"):
                    i=cell_col(c.attrib.get("r"))
                    if i is None: continue
                    typ=c.attrib.get("t")
                    v=c.find(ns+"v")
                    val=None if v is None else v.text
                    if typ=="s" and val is not None:
                        try: val=ss[int(val)]
                        except Exception: pass
                    elif typ=="inlineStr":
                        isel=c.find(ns+"is")
                        if isel is not None:
                            val="".join((t.text or "") for t in isel.iter(ns+"t"))
                    vals[i]=val
                if vals:
                    m=max(vals)
                    yield [vals.get(i) for i in range(m+1)]
                elem.clear()

def latest_stageb():
    root=BASE/"label_v1_stageB"
    cand=[]
    for d in root.glob("stageB_*"):
        p=d/"anchor_candidates_v1_stageB_LOCAL_ONLY.csv"
        r=d/"label_v1_stageB_report.txt"
        if p.is_file() and r.is_file():
            cand.append((r.stat().st_mtime,d))
    if not cand: raise SystemExit("NO_STAGE_B_WITH_CANDIDATE_CSV")
    return max(cand)[1]

def load_best_candidate(stageb):
    p=stageb/"anchor_candidates_v1_stageB_LOCAL_ONLY.csv"
    with p.open("r",encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    if not rows: raise ValueError("EMPTY_CANDIDATE_CSV")
    def score(r):
        try: return float(r.get("candidate_score") or -1)
        except Exception: return -1
    return max(rows,key=score), p

def load_panel():
    import pyarrow.parquet as pq
    if sha256(PANEL)!=PANEL_SHA256:
        raise ValueError("PANEL_HASH_MISMATCH")
    t=pq.read_table(PANEL,columns=["firm_id","year"],use_threads=False).to_pydict()
    keys={(str(f),int(y)) for f,y in zip(t["firm_id"],t["year"])}
    if len(keys)!=EXPECTED_N: raise ValueError(f"PANEL_N_UNEXPECTED:{len(keys)}")
    return keys

def q(vals,p):
    if not vals: return None
    a=sorted(vals); x=(len(a)-1)*p; lo=math.floor(x); hi=math.ceil(x)
    if lo==hi: return a[lo]
    return a[lo]*(hi-x)+a[hi]*(x-lo)

def descriptor_hits(folder: Path, field: str):
    hits=[]
    for p in sorted(folder.glob("*.txt")):
        tick()
        if p.stat().st_size>10*1024*1024: continue
        raw=None
        for enc in ("utf-8","gb18030","gbk"):
            try:
                raw=p.read_text(encoding=enc)
                break
            except Exception:
                pass
        if raw is None: continue
        lines=raw.splitlines()
        for i,line in enumerate(lines):
            if field.lower() in line.lower():
                context=" | ".join(x.strip() for x in lines[max(0,i-1):min(len(lines),i+2)] if x.strip())
                hits.append((p.name,context[:1200]))
    return hits[:20]

def reveal(p):
    if sys.platform=="darwin":
        subprocess.run(["open","-R",str(p)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage-b1",action="store_true")
    ap.add_argument("--reveal",action="store_true")
    args=ap.parse_args()
    if not args.stage_b1: ap.error("--stage-b1 required")
    os.umask(0o077)

    stageb=latest_stageb()
    best,csv_path=load_best_candidate(stageb)
    src=Path(best["path"])
    fmt=best["format"]
    if not src.is_file(): raise SystemExit(f"CANDIDATE_SOURCE_MISSING:{src}")
    if src.suffix.lower()!=".xlsx": raise SystemExit("BEST_CANDIDATE_NOT_XLSX")
    sheet=fmt.split(":",1)[1] if ":" in fmt else "sheet1"

    outroot=BASE/"label_v1_stageB1"
    outroot.mkdir(mode=0o700,exist_ok=True)
    out=outroot/("stageB1_"+datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8])
    out.mkdir(mode=0o700)
    report=out/"label_v1_stageB1_report.txt"
    anomaly=out/"anchor_anomalies_STAGEB1_LOCAL_ONLY.csv"

    print("[1/5] 核验面板与 Stage B 最佳候选……",flush=True)
    panel=load_panel()
    code_col=best.get("code_col")
    year_col=best.get("year_col") or ""
    period_col=best.get("period_col") or ""
    ann_col=best.get("announce_col")
    type_col=best.get("type_col") or ""
    if not code_col or not ann_col or (not year_col and not period_col):
        raise ValueError("BEST_CANDIDATE_COLUMNS_INCOMPLETE")

    print("[2/5] 重新只读扫描 FIN_Audit.xlsx，精确量化异常与缺失……",flush=True)
    rows=iter_xlsx_rows(src,sheet)
    headers=[("" if x is None else str(x).strip()) for x in next(rows)]
    idx={h:i for i,h in enumerate(headers)}
    needed=[code_col,ann_col]+([year_col] if year_col else [])+([period_col] if period_col else [])
    miss=[h for h in needed if h not in idx]
    if miss: raise ValueError("MISSING_COLUMNS:"+",".join(miss))

    mapping={}
    date_sets=defaultdict(set)
    raw_rows=0
    panel_matched=0
    anomalies=[]
    peryear=defaultdict(lambda: Counter())
    dates_by_year=defaultdict(list)

    for row in rows:
        tick(); raw_rows+=1
        def get(h):
            return row[idx[h]] if h and h in idx and idx[h]<len(row) else None
        code=parse_code(get(code_col))
        y=parse_year(get(year_col)) if year_col else None
        period=get(period_col) if period_col else None
        if y is None and period is not None: y=parse_year(period)
        d=parse_date(get(ann_col))
        if not code or y is None or not d or not (2010<=y<=2024): continue
        key=(code,y)
        if key not in panel: continue
        # If a report-period column exists and parses to a date, require 12/31.
        pd=parse_date(period) if period_col else None
        if pd and not (pd.month==12 and pd.day==31): continue
        panel_matched+=1
        date_sets[key].add(d)
        if key not in mapping or d<mapping[key]: mapping[key]=d

    for (code,y),d in mapping.items():
        peryear[y]["covered"]+=1
        dates_by_year[y].append(d.toordinal())
        reason=[]
        if d<=date(y,12,31): reason.append("on_or_before_fiscal_year_end")
        if d>date(y+1,12,31): reason.append("after_t_plus_1_year_end")
        if d>date(y+1,6,30): reason.append("after_june30_t_plus_1")
        if len(date_sets[(code,y)])>1: reason.append("multiple_distinct_announce_dates")
        if reason:
            anomalies.append({
                "company_code":"'"+code,
                "fiscal_year":y,
                "earliest_anchor":d.isoformat(),
                "all_distinct_dates":";".join(sorted(x.isoformat() for x in date_sets[(code,y)])),
                "reason":";".join(reason),
            })

    for code,y in panel:
        peryear[y]["panel"]+=1

    pre=sum(1 for (c,y),d in mapping.items() if d<=date(y,12,31))
    late=sum(1 for (c,y),d in mapping.items() if d>date(y+1,12,31))
    after_june=sum(1 for (c,y),d in mapping.items() if d>date(y+1,6,30))
    multi=sum(1 for k,s in date_sets.items() if len(s)>1)
    missing=len(panel)-len(mapping)

    with anomaly.open("w",encoding="utf-8",newline="") as f:
        fields=["company_code","fiscal_year","earliest_anchor","all_distinct_dates","reason"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(anomalies)

    print("[3/5] 读取同目录 CSMAR 字段说明，只提取 Annodt/日期字段语义……",flush=True)
    hits=descriptor_hits(src.parent,ann_col)
    semantic_text=" ".join(x[1] for x in hits)
    mentions_announcement=bool(re.search(r"公告|披露|发布|公布|announcement|announce|disclosure|publish",semantic_text,re.I))
    mentions_audit_report_date=bool(re.search(r"审计报告日|审计报告日期|audit report date|签字日期|签署日期",semantic_text,re.I))
    mentions_annual=bool(re.search(r"年度报告|年报|annual report|报告期",semantic_text,re.I))

    print("[4/5] 生成逐年覆盖与日期分布汇总……",flush=True)
    year_lines=[]
    for y in range(2010,2025):
        panel_n=peryear[y]["panel"]; cov=peryear[y]["covered"]
        ords=dates_by_year[y]
        med=date.fromordinal(round(q(ords,.5))).isoformat() if ords else "NA"
        p05=date.fromordinal(round(q(ords,.05))).isoformat() if ords else "NA"
        p95=date.fromordinal(round(q(ords,.95))).isoformat() if ords else "NA"
        year_lines.append(f"{y}: panel={panel_n}, anchors={cov}, coverage={cov/panel_n:.6f}, date_p05={p05}, median={med}, date_p95={p95}")

    print("[5/5] 写出可上传报告；不授权最终标签……",flush=True)
    lines=[
        "PeerJ 141707 | Label v1 Stage B1 anchor-source adjudication",
        "Purpose: decide whether FIN_Audit Annodt can serve as annual-report first-public filing anchor",
        "NO FINAL LABELS; NO TRAINING",
        "Local time: "+datetime.now().astimezone().isoformat(),
        "Stage B: "+str(stageb),
        "Output: "+str(out),
        "",
        "A. SOURCE / FIELD",
        "source="+str(src),
        "source_sha256="+sha256(src),
        "sheet="+sheet,
        "code_col="+str(code_col),
        "year_col="+str(year_col),
        "period_col="+str(period_col),
        "announce_col="+str(ann_col),
        "type_col="+str(type_col),
        "",
        "B. EXACT PANEL COVERAGE / ANOMALIES",
        f"panel_n={len(panel)}",
        f"unique_anchor_keys={len(mapping)}",
        f"missing_anchor_keys={missing}",
        f"coverage={len(mapping)/len(panel):.8f}",
        f"raw_panel_matched_rows_before_earliest_dedup={panel_matched}",
        f"firm_years_with_multiple_distinct_announce_dates={multi}",
        f"anchor_on_or_before_fiscal_year_end={pre}",
        f"anchor_after_t_plus_1_year_end={late}",
        f"anchor_after_june30_t_plus_1={after_june}",
        f"local_anomaly_rows={len(anomalies)}",
        "",
        "C. DESCRIPTOR SEMANTIC CHECK",
        f"descriptor_hits_for_{ann_col}={len(hits)}",
        f"mentions_announcement_or_disclosure_semantics={str(mentions_announcement).lower()}",
        f"mentions_explicit_audit_report_date_or_signature_semantics={str(mentions_audit_report_date).lower()}",
        f"mentions_annual_report_or_reporting_period_semantics={str(mentions_annual).lower()}",
    ]
    for i,(fn,ctx) in enumerate(hits[:8],1):
        # Metadata only; no company-level records.
        lines.append(f"descriptor_hit_{i}={fn} :: {ctx}")
    lines += ["","D. YEAR-BY-YEAR COVERAGE / DATE DISTRIBUTION",*year_lines]
    lines += [
        "",
        "E. DECISION GATE",
        "No automatic semantic approval is made in Stage B1.",
        "Use this report to decide whether the field is truly a public annual-report filing/announcement date rather than an audit-report signing date.",
        "Anomalous company-year rows remain LOCAL_ONLY and are not adjudicated here.",
        "",
        "source_files_modified=false",
        "training_or_prediction_run=false",
        "final_nonnull_labels_written=0",
        "STATUS: LABEL_V1_STAGE_B1_COMPLETE_NEEDS_SEMANTIC_ADJUDICATION",
        "Upload only label_v1_stageB1_report.txt.",
    ]
    report.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("STATUS: LABEL_V1_STAGE_B1_COMPLETE_NEEDS_SEMANTIC_ADJUDICATION")
    print("请只上传：",report)
    if args.reveal: reveal(report)

if __name__=="__main__":
    main()
