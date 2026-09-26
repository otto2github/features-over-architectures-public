#!/usr/bin/env python3
"""
PeerJ 141707 — Label v1 Stage B3
Build a public annual-report disclosure-date anchor from CNINFO announcement metadata.

Purpose
-------
Stage B2 established that no suitable local public-disclosure anchor exists.
Stage B3 retrieves ONLY announcement metadata for annual reports from CNINFO,
then maps the earliest eligible full annual-report announcement to the fixed
51,675 firm-year panel.

Safety / provenance
-------------------
- No PDF download.
- No model training.
- No final fraud labels.
- No source overwrite.
- Single-threaded, polite network access; default delay 1.2 s/request.
- Resume cache under ~/peerj_141707_audit/cninfo_public_anchor_cache_v1/
- Outputs under ~/peerj_141707_audit/label_v1_stageB3/
- FIN_Audit.Annodt is NOT used.
- On macOS, --reveal opens Finder and selects the final report.

The script first performs a two-exchange probe. If the endpoint/response shape
is not as expected, it stops safely before a full crawl.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path.home() / "peerj_141707_audit"
PROJECT = Path(os.environ.get("PEERJ_SOURCE_PROJECT_ROOT", "private_inputs/project"))
PANEL = PROJECT / "data/processed/kg/node_features_v1_1.parquet"
PANEL_SHA256 = "adc00129f06ffb0995691c6d1ba98f740c53561c1e0cc4e91d927b84eb36beb2"

OUT_ROOT = BASE / "label_v1_stageB3"
CACHE_ROOT = BASE / "cninfo_public_anchor_cache_v1"

QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
REFERER = "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search"
CATEGORY = "category_ndbg_szsh;"
PAGE_SIZE = 30
BEIJING = ZoneInfo("Asia/Shanghai")

# Fiscal years in the fixed panel are 2010–2024; their annual reports normally
# become public in calendar years 2011–2025. We query the whole calendar year
# so that delayed reports are not silently dropped.
FISCAL_YEARS = list(range(2010, 2025))
EXCHANGES = ("sse", "szse")

FULL_REPORT_RE = re.compile(
    r"(?P<year>20\d{2})\s*(?:年\s*)?(?:年度报告|年报)",
    re.I,
)
SUMMARY_RE = re.compile(r"摘要|summary", re.I)
ENGLISH_RE = re.compile(r"英文|english", re.I)
NON_REPORT_RE = re.compile(
    r"问询|回复|意见|说明|公告|提示|预约|审计报告|鉴证报告|审核报告|社会责任|"
    r"环境社会|ESG|可持续|内部控制|董事会|监事会",
    re.I,
)
REVISED_RE = re.compile(r"修订|更正|更新|补充|修正版|修订版", re.I)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def load_panel():
    import pyarrow.parquet as pq
    if sha256(PANEL) != PANEL_SHA256:
        raise ValueError("PANEL_HASH_MISMATCH")
    d = pq.read_table(PANEL, columns=["firm_id", "year"], use_threads=False).to_pydict()
    keys = {(str(f), int(y)) for f, y in zip(d["firm_id"], d["year"])}
    if len(keys) != 51675:
        raise ValueError(f"PANEL_COUNT_UNEXPECTED:{len(keys)}")
    return keys

def form_payload(column: str, page: int, start: str, end: str):
    return {
        "pageNum": str(page),
        "pageSize": str(PAGE_SIZE),
        "tabName": "fulltext",
        "column": column,
        "stock": "",
        "searchkey": "",
        "secid": "",
        "plate": "",
        "category": CATEGORY,
        "trade": "",
        "seDate": f"{start}~{end}",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }

def request_json(payload, timeout=25, retries=4):
    data = urllib.parse.urlencode(payload).encode("utf-8")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/153.0.0.0 Safari/537.36"
        ),
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://www.cninfo.com.cn",
        "Referer": REFERER,
        "Accept": "application/json, text/javascript, */*; q=0.01",
    }
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(QUERY_URL, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP_{resp.status}")
                obj = json.loads(raw.decode("utf-8", errors="strict"))
                return obj, raw
        except Exception as e:
            last = e
            if attempt + 1 < retries:
                time.sleep(min(12, 1.5 * (2 ** attempt)) + random.uniform(0.1, 0.6))
    raise RuntimeError(f"CNINFO_REQUEST_FAILED:{type(last).__name__}:{last}")

def ann_date_from_epoch_ms(v):
    try:
        ms = int(v)
    except Exception:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).astimezone(BEIJING).date()
    except Exception:
        return None

def clean_title(s):
    if s is None:
        return ""
    s = re.sub(r"</?em>", "", str(s), flags=re.I)
    s = re.sub(r"\s+", "", s)
    return s

def classify_title(title):
    t = clean_title(title)
    m = FULL_REPORT_RE.search(t)
    if not m:
        return None, "not_annual_report_title"
    fy = int(m.group("year"))
    if SUMMARY_RE.search(t):
        return fy, "summary"
    if ENGLISH_RE.search(t):
        return fy, "english"
    if NON_REPORT_RE.search(t):
        return fy, "related_nonreport"
    if REVISED_RE.search(t):
        return fy, "full_revised"
    return fy, "full_original"

def adjunct_path_date(url):
    if not url:
        return None
    m = re.search(r"(?:^|/)finalpage/(\d{4})-(\d{2})-(\d{2})/", str(url))
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None

def cache_page_path(column, fiscal_year, page):
    d = CACHE_ROOT / column / str(fiscal_year)
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    return d / f"page_{page:04d}.json"

def read_or_fetch_page(column, fiscal_year, page, delay):
    path = cache_page_path(column, fiscal_year, page)
    if path.is_file():
        raw = path.read_bytes()
        return json.loads(raw.decode("utf-8")), "cache"

    cal_year = fiscal_year + 1
    payload = form_payload(column, page, f"{cal_year}-01-01", f"{cal_year}-12-31")
    obj, raw = request_json(payload)

    # Only cache responses having the expected basic shape.
    if not isinstance(obj, dict) or "announcements" not in obj:
        raise RuntimeError("CNINFO_RESPONSE_SHAPE_UNEXPECTED")
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)

    time.sleep(max(0.0, delay) + random.uniform(0.0, 0.25))
    return obj, "network"

def response_page_count(obj):
    for k in ("totalpages", "totalPages"):
        try:
            n = int(obj.get(k))
            if n >= 1:
                return n
        except Exception:
            pass
    try:
        total = int(obj.get("totalRecordNum") or 0)
        if total > 0:
            return math.ceil(total / PAGE_SIZE)
    except Exception:
        pass
    anns = obj.get("announcements")
    return 1 if isinstance(anns, list) else 0

def safe_ann_list(obj):
    a = obj.get("announcements")
    return a if isinstance(a, list) else []

def validate_probe(obj, column):
    anns = safe_ann_list(obj)
    if not anns:
        return False, f"{column}:empty_announcements"
    sample = anns[: min(10, len(anns))]
    required_hits = 0
    for x in sample:
        if isinstance(x, dict) and x.get("secCode") and x.get("announcementTitle") and x.get("announcementTime") is not None:
            required_hits += 1
    if required_hits < max(1, len(sample) // 2):
        return False, f"{column}:required_fields_missing"
    return True, "ok"

def select_anchor(candidates):
    # Earliest original full report. If no original title is present, use the
    # earliest revised full report but mark it fallback.
    originals = [x for x in candidates if x["title_class"] == "full_original"]
    if originals:
        x = min(originals, key=lambda z: z["announcement_date"])
        return x, "full_original_earliest"
    revised = [x for x in candidates if x["title_class"] == "full_revised"]
    if revised:
        x = min(revised, key=lambda z: z["announcement_date"])
        return x, "full_revised_fallback"
    return None, "no_full_report"

def write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")

def reveal(path):
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(path)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-b3", action="store_true")
    ap.add_argument("--delay", type=float, default=1.2,
                    help="seconds between network requests; default 1.2")
    ap.add_argument("--reveal", action="store_true")
    args = ap.parse_args()
    if not args.stage_b3:
        ap.error("--stage-b3 required")
    if args.delay < 0.8:
        ap.error("--delay must be >= 0.8 seconds")

    os.umask(0o077)
    OUT_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    out = OUT_ROOT / ("stageB3_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    out.mkdir(mode=0o700)
    report = out / "label_v1_stageB3_report.txt"

    try:
        print("[1/6] 核验固定 51,675 firm-year 面板……", flush=True)
        panel = load_panel()

        print("[2/6] 小规模探测 CNINFO 年报公告元数据接口（不下载PDF）……", flush=True)
        probe_info = {}
        for col in EXCHANGES:
            obj, source = read_or_fetch_page(col, 2023, 1, args.delay)
            ok, why = validate_probe(obj, col)
            probe_info[col] = {
                "ok": ok,
                "reason": why,
                "source": source,
                "records": len(safe_ann_list(obj)),
                "pages": response_page_count(obj),
            }

        if not all(x["ok"] for x in probe_info.values()):
            lines = [
                "PeerJ 141707 | Label v1 Stage B3 CNINFO public-anchor retrieval",
                "NO FINAL LABELS; NO TRAINING; NO PDF DOWNLOAD",
                "Local time: " + datetime.now().astimezone().isoformat(),
                "",
                "A. PROBE",
                json.dumps(probe_info, ensure_ascii=False, sort_keys=True),
                "",
                "B. DECISION",
                "full_retrieval_started=false",
                "The CNINFO metadata endpoint did not pass the response-shape probe for both exchanges.",
                "No public filing anchor was inferred.",
                "",
                "source_files_modified=false",
                "training_or_prediction_run=false",
                "final_nonnull_labels_written=0",
                "STATUS: LABEL_V1_STAGE_B3_STOP_CNINFO_PROBE_FAILED",
            ]
            report.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(lines[-1])
            print("请上传：", report)
            if args.reveal:
                reveal(report)
            return

        print("[3/6] 单线程、可续传获取 2010–2024 财年年报公告元数据……", flush=True)
        by_key = defaultdict(list)
        raw_count = 0
        accepted_title_count = 0
        request_stats = Counter()
        exchange_year_stats = []

        for fy in FISCAL_YEARS:
            for col in EXCHANGES:
                obj, src = read_or_fetch_page(col, fy, 1, args.delay)
                request_stats[src] += 1
                pages = response_page_count(obj)
                if pages <= 0 or pages > 1000:
                    raise RuntimeError(f"UNEXPECTED_PAGE_COUNT:{col}:{fy}:{pages}")

                year_raw = 0
                year_kept = 0
                for page in range(1, pages + 1):
                    if page == 1:
                        cur = obj
                    else:
                        cur, src2 = read_or_fetch_page(col, fy, page, args.delay)
                        request_stats[src2] += 1

                    for ann in safe_ann_list(cur):
                        if not isinstance(ann, dict):
                            continue
                        raw_count += 1
                        year_raw += 1
                        code = str(ann.get("secCode") or "").strip()
                        if not re.fullmatch(r"\d{6}", code):
                            continue
                        title = clean_title(ann.get("announcementTitle"))
                        title_fy, title_class = classify_title(title)
                        if title_fy != fy:
                            continue
                        ann_date = ann_date_from_epoch_ms(ann.get("announcementTime"))
                        if not ann_date:
                            continue
                        key = (code, fy)
                        if key not in panel:
                            continue

                        accepted_title_count += 1
                        year_kept += 1
                        adj = ann.get("adjunctUrl")
                        path_date = adjunct_path_date(adj)
                        by_key[key].append({
                            "company_code": code,
                            "fiscal_year": fy,
                            "exchange_query": col,
                            "announcement_date": ann_date,
                            "announcement_time_ms": str(ann.get("announcementTime")),
                            "announcement_title": title,
                            "title_class": title_class,
                            "announcement_id": str(ann.get("announcementId") or ""),
                            "org_id": str(ann.get("orgId") or ""),
                            "adjunct_url": str(adj or ""),
                            "adjunct_path_date": path_date,
                            "path_date_matches": (path_date == ann_date) if path_date else None,
                        })

                exchange_year_stats.append({
                    "fiscal_year": fy,
                    "calendar_query_year": fy + 1,
                    "column": col,
                    "pages": pages,
                    "raw_records": year_raw,
                    "panel_title_matched_records": year_kept,
                })
                print(f"      FY{fy} {col}: pages={pages}, raw={year_raw}, panel/title matched={year_kept}", flush=True)

        print("[4/6] 按 firm-year 选择最早完整年度报告公告日期……", flush=True)
        anchor_rows = []
        class_counts = Counter()
        missing_by_year = Counter()
        covered_by_year = Counter()
        path_check = Counter()
        suspicious = []

        for code, fy in sorted(panel, key=lambda x: (x[1], x[0])):
            cand = by_key.get((code, fy), [])
            chosen, rule = select_anchor(cand)
            if chosen is None:
                missing_by_year[fy] += 1
                anchor_rows.append({
                    "company_code": code,
                    "fiscal_year": fy,
                    "annual_report_public_anchor": None,
                    "anchor_rule": "missing",
                    "announcement_id": None,
                    "announcement_title": None,
                    "adjunct_url": None,
                    "source": "CNINFO annual-report announcement metadata",
                })
                continue

            d = chosen["announcement_date"]
            covered_by_year[fy] += 1
            class_counts[rule] += 1
            if chosen["path_date_matches"] is True:
                path_check["match"] += 1
            elif chosen["path_date_matches"] is False:
                path_check["mismatch"] += 1
            else:
                path_check["no_path_date"] += 1

            # Timing plausibility flag only; never alter the date.
            timing_flag = "normal_t_plus_1"
            if d <= date(fy, 12, 31):
                timing_flag = "on_or_before_fiscal_year_end"
            elif d > date(fy + 1, 12, 31):
                timing_flag = "after_t_plus_1_year_end"

            if timing_flag != "normal_t_plus_1":
                suspicious.append({
                    "company_code": "'" + code,
                    "fiscal_year": fy,
                    "announcement_date": d.isoformat(),
                    "timing_flag": timing_flag,
                    "announcement_title": chosen["announcement_title"],
                    "announcement_id": chosen["announcement_id"],
                })

            anchor_rows.append({
                "company_code": code,
                "fiscal_year": fy,
                "annual_report_public_anchor": d.isoformat(),
                "anchor_rule": rule,
                "timing_flag": timing_flag,
                "announcement_id": chosen["announcement_id"],
                "announcement_title": chosen["announcement_title"],
                "adjunct_url": chosen["adjunct_url"],
                "announcement_time_ms": chosen["announcement_time_ms"],
                "exchange_query": chosen["exchange_query"],
                "source": "CNINFO annual-report announcement metadata",
            })

        write_parquet(out / "annual_report_public_anchors_CNINFO_v1.parquet", anchor_rows)

        with (out / "anchor_anomalies_STAGEB3_LOCAL_ONLY.csv").open("w", encoding="utf-8", newline="") as f:
            fields = ["company_code","fiscal_year","announcement_date","timing_flag","announcement_title","announcement_id"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(suspicious)

        with (out / "exchange_year_request_summary.csv").open("w", encoding="utf-8", newline="") as f:
            fields = ["fiscal_year","calendar_query_year","column","pages","raw_records","panel_title_matched_records"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(exchange_year_stats)

        print("[5/6] 计算覆盖率与可复现性检查……", flush=True)
        n_cov = sum(covered_by_year.values())
        coverage = n_cov / len(panel)
        normal = sum(
            1 for r in anchor_rows
            if r.get("annual_report_public_anchor") and r.get("timing_flag") == "normal_t_plus_1"
        )
        original_n = class_counts["full_original_earliest"]
        revised_fallback_n = class_counts["full_revised_fallback"]

        year_lines = []
        for fy in FISCAL_YEARS:
            total = sum(1 for c,y in panel if y == fy)
            cov = covered_by_year[fy]
            year_lines.append(
                f"{fy}: panel={total}, anchors={cov}, missing={missing_by_year[fy]}, coverage={cov/total:.6f}"
            )

        print("[6/6] 写出可上传报告（仍不生成最终 fraud 标签）……", flush=True)
        lines = [
            "PeerJ 141707 | Label v1 Stage B3 CNINFO public annual-report anchor",
            "NO FINAL LABELS; NO TRAINING; NO PDF DOWNLOAD",
            "Local time: " + datetime.now().astimezone().isoformat(),
            "Output: " + str(out),
            "",
            "A. SOURCE / PROBE",
            "source=CNINFO announcement metadata",
            "query_url=" + QUERY_URL,
            "annual_report_category=" + CATEGORY,
            "probe=" + json.dumps(probe_info, ensure_ascii=False, sort_keys=True),
            "FIN_Audit_Annodt_used=false",
            "",
            "B. RETRIEVAL",
            f"raw_announcement_records_seen={raw_count}",
            f"panel_and_fiscal_year_title_matched_records={accepted_title_count}",
            "request_sources=" + json.dumps(dict(request_stats), ensure_ascii=False, sort_keys=True),
            f"polite_delay_seconds={args.delay}",
            "concurrency=1",
            "pdf_downloads=0",
            "",
            "C. PANEL ANCHOR COVERAGE",
            f"panel_n={len(panel)}",
            f"anchor_n={n_cov}",
            f"missing_anchor_n={len(panel)-n_cov}",
            f"coverage={coverage:.8f}",
            f"normal_t_plus_1_anchor_n={normal}",
            f"full_original_earliest_n={original_n}",
            f"full_revised_fallback_n={revised_fallback_n}",
            f"timing_anomaly_n={len(suspicious)}",
            "adjunct_path_date_crosscheck=" + json.dumps(dict(path_check), ensure_ascii=False, sort_keys=True),
            "",
            "D. YEAR-BY-YEAR COVERAGE",
            *year_lines,
            "",
            "E. DECISION GATE",
        ]

        # This is a provenance gate, not a final-label gate.
        if coverage >= 0.95 and normal / max(1, n_cov) >= 0.99:
            lines += [
                "public_anchor_source_eligible_for_freeze=true",
                "The CNINFO public announcement metadata provides sufficient panel coverage and timing plausibility for the next post-filing integration stage.",
                "Missing/anomalous firm-years remain unresolved and must not be auto-filled from audit dates or filesystem timestamps.",
                "STATUS: LABEL_V1_STAGE_B3_PUBLIC_ANCHOR_READY_FOR_FREEZE",
            ]
        else:
            lines += [
                "public_anchor_source_eligible_for_freeze=false",
                "Coverage/timing gate was not met. No post-filing status should be inferred yet.",
                "STATUS: LABEL_V1_STAGE_B3_STOP_COVERAGE_OR_TIMING_GATE",
            ]

        lines += [
            "",
            "source_files_modified=false",
            "training_or_prediction_run=false",
            "final_nonnull_labels_written=0",
            "Upload only label_v1_stageB3_report.txt.",
        ]
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")

        manifest = {
            "created_at": datetime.now().astimezone().isoformat(),
            "panel_sha256": PANEL_SHA256,
            "query_url": QUERY_URL,
            "category": CATEGORY,
            "delay_seconds": args.delay,
            "concurrency": 1,
            "pdf_downloads": 0,
            "coverage": coverage,
            "anchor_output_sha256": sha256(out / "annual_report_public_anchors_CNINFO_v1.parquet"),
            "final_labels_authorized": False,
            "training_authorized": False,
        }
        (out / "manifest_v1_stageB3.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8"
        )

        status = [x for x in lines if x.startswith("STATUS:")][0]
        print(status)
        print("请只上传：", report)
        if args.reveal:
            reveal(report)

    except Exception as e:
        report.write_text(
            "\n".join([
                "PeerJ 141707 | Label v1 Stage B3 FAILED SAFELY",
                "Local time: " + datetime.now().astimezone().isoformat(),
                f"ERROR_TYPE={type(e).__name__}",
                f"ERROR={e}",
                "source_files_modified=false",
                "training_or_prediction_run=false",
                "final_nonnull_labels_written=0",
                "STATUS: LABEL_V1_STAGE_B3_FAILED_SAFE",
            ]) + "\n",
            encoding="utf-8"
        )
        print(f"Stage B3 failed safely: {type(e).__name__}: {e}", file=sys.stderr)
        print("请上传：", report, file=sys.stderr)
        if args.reveal:
            reveal(report)
        raise

if __name__ == "__main__":
    main()
