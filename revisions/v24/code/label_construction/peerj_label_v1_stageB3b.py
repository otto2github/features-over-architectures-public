#!/usr/bin/env python3
"""PeerJ 141707 — Label v1 Stage B3b.

Corrected CNINFO annual-report public-anchor retrieval with exchange+plate routing.
B3 left `plate` blank, so its ~39% coverage result is superseded and must not be
used as evidence of CNINFO coverage.

Reads the fixed 51,675 firm-year panel, retrieves CNINFO annual-report announcement
metadata only (no PDFs), and writes only under ~/peerj_141707_audit.
No model training and no final fraud labels.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, re, subprocess, sys, time
import urllib.parse, urllib.request, uuid
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path.home() / "peerj_141707_audit"
PROJECT = Path(os.environ.get("PEERJ_SOURCE_PROJECT_ROOT", "private_inputs/project"))
PANEL = PROJECT / "data/processed/kg/node_features_v1_1.parquet"
PANEL_SHA256 = "adc00129f06ffb0995691c6d1ba98f740c53561c1e0cc4e91d927b84eb36beb2"
OUT_ROOT = BASE / "label_v1_stageB3b"
CACHE_ROOT = BASE / "cninfo_public_anchor_cache_v2"
QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
REFERER = "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search"
CATEGORY = "category_ndbg_szsh;"
PAGE_SIZE = 30
BJ = ZoneInfo("Asia/Shanghai")
FISCAL_YEARS = list(range(2010, 2025))

# Current CNINFO usage requires column+plate to be paired.
ROUTES = (
    {"name":"sse_main", "column":"sse",  "plate":"sh",    "prefix":re.compile(r"^6(?!8)")},
    {"name":"sse_star", "column":"sse",  "plate":"shkcp", "prefix":re.compile(r"^68")},
    {"name":"szse_main","column":"szse", "plate":"sz",    "prefix":re.compile(r"^00")},
    {"name":"szse_gem", "column":"szse", "plate":"szcy",  "prefix":re.compile(r"^30")},
)

FULL_RE = re.compile(r"(?P<year>20\d{2})\s*(?:年\s*)?(?:年度报告|年报)", re.I)
SUMMARY_RE = re.compile(r"摘要|summary", re.I)
ENGLISH_RE = re.compile(r"英文|english", re.I)
RELATED_RE = re.compile(r"问询|回复|说明|提示|预约|审计报告|鉴证报告|审核报告|社会责任|ESG|可持续|内部控制", re.I)
REVISED_RE = re.compile(r"修订|更正|更新|补充|修正版|修订版", re.I)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def load_panel():
    import pyarrow.parquet as pq
    if sha256(PANEL) != PANEL_SHA256:
        raise ValueError("PANEL_HASH_MISMATCH")
    d = pq.read_table(PANEL, columns=["firm_id","year"], use_threads=False).to_pydict()
    keys = {(str(f), int(y)) for f,y in zip(d["firm_id"], d["year"])}
    if len(keys) != 51675:
        raise ValueError(f"PANEL_COUNT_UNEXPECTED:{len(keys)}")
    return keys


def payload(route, page, fiscal_year):
    y = fiscal_year + 1
    return {
        "pageNum":str(page), "pageSize":str(PAGE_SIZE), "tabName":"fulltext",
        "column":route["column"], "stock":"", "searchkey":"", "secid":"",
        "plate":route["plate"], "category":CATEGORY, "trade":"",
        "seDate":f"{y}-01-01~{y}-12-31", "sortName":"", "sortType":"",
        "isHLtitle":"true",
    }


def request_json(data, timeout=30, retries=4):
    body = urllib.parse.urlencode(data).encode("utf-8")
    headers = {
        "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/153.0 Safari/537.36",
        "Content-Type":"application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With":"XMLHttpRequest", "Origin":"https://www.cninfo.com.cn",
        "Referer":REFERER, "Accept":"application/json, text/javascript, */*; q=0.01",
    }
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(QUERY_URL, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                return json.loads(raw.decode("utf-8")), raw
        except Exception as e:
            last = e
            if i+1 < retries:
                time.sleep(min(12, 1.5*(2**i)) + random.uniform(.1,.5))
    raise RuntimeError(f"CNINFO_REQUEST_FAILED:{type(last).__name__}:{last}")


def cache_path(route, fy, page):
    d = CACHE_ROOT / route["name"] / str(fy)
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    return d / f"page_{page:04d}.json"


def fetch_page(route, fy, page, delay):
    p = cache_path(route, fy, page)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8")), "cache"
    obj, raw = request_json(payload(route, page, fy))
    if not isinstance(obj, dict) or not isinstance(obj.get("announcements"), list):
        raise RuntimeError(f"CNINFO_RESPONSE_SHAPE_UNEXPECTED:{route['name']}:{fy}:{page}")
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, p)
    time.sleep(delay + random.uniform(0,.2))
    return obj, "network"


def pages(obj):
    for k in ("totalpages","totalPages"):
        try:
            n = int(obj.get(k))
            if n >= 1: return n
        except Exception: pass
    try:
        n = int(obj.get("totalRecordNum") or 0)
        if n > 0: return math.ceil(n/PAGE_SIZE)
    except Exception: pass
    return 1 if obj.get("announcements") else 0


def clean_title(x):
    return re.sub(r"\s+", "", re.sub(r"</?em>", "", str(x or ""), flags=re.I))


def classify(title):
    t = clean_title(title)
    m = FULL_RE.search(t)
    if not m: return None, "not_annual"
    fy = int(m.group("year"))
    if SUMMARY_RE.search(t): return fy, "summary"
    if ENGLISH_RE.search(t): return fy, "english"
    if RELATED_RE.search(t): return fy, "related_nonreport"
    if REVISED_RE.search(t): return fy, "full_revised"
    return fy, "full_original"


def ann_date(ms):
    try:
        return datetime.fromtimestamp(int(ms)/1000, tz=timezone.utc).astimezone(BJ).date()
    except Exception:
        return None


def adjunct_date(url):
    m = re.search(r"(?:^|/)finalpage/(\d{4})-(\d{2})-(\d{2})/", str(url or ""))
    if not m: return None
    try: return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError: return None


def route_ok(route, code):
    return bool(route["prefix"].search(code or ""))


def probe(route, fy, delay):
    obj, src = fetch_page(route, fy, 1, delay)
    a = [x for x in (obj.get("announcements") or [])[:30] if isinstance(x,dict)]
    codes = [str(x.get("secCode") or "") for x in a if x.get("secCode")]
    purity = sum(route_ok(route,c) for c in codes)/len(codes) if codes else 0.0
    required = sum(bool(x.get("secCode")) and bool(x.get("announcementTitle")) and x.get("announcementTime") is not None for x in a)
    return {"ok":bool(a) and required >= max(1,len(a)//2) and purity >= .80,
            "source":src, "records":len(a), "pages":pages(obj),
            "sample_code_n":len(codes), "route_prefix_purity":purity}


def choose(cands):
    x = [a for a in cands if a["title_class"] == "full_original"]
    if x: return min(x, key=lambda z:z["announcement_date"]), "full_original_earliest"
    x = [a for a in cands if a["title_class"] == "full_revised"]
    if x: return min(x, key=lambda z:z["announcement_date"]), "full_revised_fallback"
    return None, "missing"


def write_parquet(path, rows):
    import pyarrow as pa, pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def reveal(path):
    if sys.platform == "darwin":
        subprocess.run(["open","-R",str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-b3b", action="store_true")
    ap.add_argument("--delay", type=float, default=1.2)
    ap.add_argument("--reveal", action="store_true")
    args = ap.parse_args()
    if not args.stage_b3b: ap.error("--stage-b3b required")
    if args.delay < .8: ap.error("--delay must be >= 0.8")
    os.umask(0o077)

    OUT_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    out = OUT_ROOT / ("stageB3b_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    out.mkdir(mode=0o700)
    report = out / "label_v1_stageB3b_report.txt"

    try:
        print("[1/6] 核验固定面板与代码前缀……", flush=True)
        panel = load_panel()
        prefixes = Counter()
        for c,y in panel:
            if c.startswith("68"): prefixes["sse_star"] += 1
            elif c.startswith("6"): prefixes["sse_main"] += 1
            elif c.startswith("30"): prefixes["szse_gem"] += 1
            elif c.startswith("00"): prefixes["szse_main"] += 1
            elif c.startswith(("4","8")): prefixes["bse_or_neeq"] += 1
            else: prefixes["other"] += 1

        print("[2/6] 探测正确 column+plate 路由……", flush=True)
        probes = {}
        for r in ROUTES:
            probes[r["name"]] = probe(r, 2023, args.delay)
            p = probes[r["name"]]
            print(f"      {r['name']}: ok={p['ok']} pages={p['pages']} purity={p['route_prefix_purity']:.1%}", flush=True)
        if not all(x["ok"] for x in probes.values()):
            lines=[
                "PeerJ 141707 | Label v1 Stage B3b corrected CNINFO routing",
                "NO FINAL LABELS; NO TRAINING; NO PDF DOWNLOAD",
                "Local time="+datetime.now().astimezone().isoformat(), "",
                "A. PANEL PREFIX COUNTS", json.dumps(dict(prefixes),ensure_ascii=False,sort_keys=True), "",
                "B. ROUTE PROBES", json.dumps(probes,ensure_ascii=False,sort_keys=True), "",
                "C. DECISION", "full_crawl_started=false",
                "At least one corrected route failed purity/shape validation.",
                "STATUS: LABEL_V1_STAGE_B3B_STOP_ROUTE_PROBE_FAILED", "",
                "source_files_modified=false", "training_or_prediction_run=false", "final_nonnull_labels_written=0"]
            report.write_text("\n".join(lines)+"\n",encoding="utf-8")
            print("STATUS: LABEL_V1_STAGE_B3B_STOP_ROUTE_PROBE_FAILED")
            print("请上传：",report)
            if args.reveal: reveal(report)
            return

        print("[3/6] 获取 2010–2024 财年年报公告元数据（不下载PDF）……", flush=True)
        by_key = defaultdict(list); req = Counter(); route_year=[]; raw_seen=0; matched=0
        for fy in FISCAL_YEARS:
            for r in ROUTES:
                obj,src = fetch_page(r,fy,1,args.delay); req[src]+=1
                n_pages = pages(obj)
                if n_pages <= 0 or n_pages > 1000:
                    raise RuntimeError(f"PAGE_COUNT_UNEXPECTED:{r['name']}:{fy}:{n_pages}")
                yr_raw=0; yr_match=0
                for pg in range(1,n_pages+1):
                    cur = obj if pg==1 else fetch_page(r,fy,pg,args.delay)[0]
                    if pg>1: req["network_or_cache"] += 1
                    for a in cur.get("announcements") or []:
                        if not isinstance(a,dict): continue
                        raw_seen += 1; yr_raw += 1
                        code = str(a.get("secCode") or "").strip()
                        if not re.fullmatch(r"\d{6}",code) or not route_ok(r,code): continue
                        tfy,tclass = classify(a.get("announcementTitle"))
                        if tfy != fy: continue
                        d = ann_date(a.get("announcementTime"))
                        if not d or (code,fy) not in panel: continue
                        pd = adjunct_date(a.get("adjunctUrl"))
                        matched += 1; yr_match += 1
                        by_key[(code,fy)].append({
                            "announcement_date":d, "title_class":tclass,
                            "announcement_title":clean_title(a.get("announcementTitle")),
                            "announcement_id":str(a.get("announcementId") or ""),
                            "adjunct_url":str(a.get("adjunctUrl") or ""),
                            "path_date_matches": (pd==d) if pd else None,
                            "route":r["name"]})
                route_year.append({"fiscal_year":fy,"calendar_query_year":fy+1,"route":r["name"],
                                   "pages":n_pages,"raw_records":yr_raw,"panel_title_matched":yr_match})
                print(f"      FY{fy} {r['name']}: pages={n_pages}, raw={yr_raw}, matched={yr_match}", flush=True)

        print("[4/6] 聚合为 firm-year 最早完整年报公开公告日期……", flush=True)
        anchors=[]; covered=Counter(); rules=Counter(); pathcheck=Counter(); anomalies=[]
        for code,fy in sorted(panel,key=lambda z:(z[1],z[0])):
            chosen,rule = choose(by_key.get((code,fy),[]))
            if not chosen:
                anchors.append({"company_code":code,"fiscal_year":fy,"annual_report_public_anchor":None,
                                "anchor_rule":"missing","source":"CNINFO corrected exchange+plate routing"})
                continue
            d = chosen["announcement_date"]; covered[fy]+=1; rules[rule]+=1
            if chosen["path_date_matches"] is True: pathcheck["match"]+=1
            elif chosen["path_date_matches"] is False: pathcheck["mismatch"]+=1
            else: pathcheck["no_path_date"]+=1
            timing = "normal_t_plus_1"
            if d <= date(fy,12,31): timing="on_or_before_fiscal_year_end"
            elif d > date(fy+1,12,31): timing="after_t_plus_1_year_end"
            if timing != "normal_t_plus_1":
                anomalies.append({"company_code":"'"+code,"fiscal_year":fy,"announcement_date":d.isoformat(),
                                  "timing_flag":timing,"announcement_title":chosen["announcement_title"],
                                  "announcement_id":chosen["announcement_id"]})
            anchors.append({"company_code":code,"fiscal_year":fy,"annual_report_public_anchor":d.isoformat(),
                            "anchor_rule":rule,"timing_flag":timing,"announcement_id":chosen["announcement_id"],
                            "announcement_title":chosen["announcement_title"],"adjunct_url":chosen["adjunct_url"],
                            "route":chosen["route"],"source":"CNINFO corrected exchange+plate routing"})

        write_parquet(out/"annual_report_public_anchors_CNINFO_v2.parquet", anchors)
        with (out/"anchor_anomalies_STAGEB3B_LOCAL_ONLY.csv").open("w",encoding="utf-8",newline="") as f:
            fields=["company_code","fiscal_year","announcement_date","timing_flag","announcement_title","announcement_id"]
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(anomalies)
        with (out/"route_year_summary.csv").open("w",encoding="utf-8",newline="") as f:
            fields=["fiscal_year","calendar_query_year","route","pages","raw_records","panel_title_matched"]
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(route_year)

        print("[5/6] 计算覆盖率与剩余代码前缀……", flush=True)
        n_cov=sum(covered.values()); coverage=n_cov/len(panel)
        normal=sum(1 for x in anchors if x.get("annual_report_public_anchor") and x.get("timing_flag")=="normal_t_plus_1")
        yearlines=[]
        for fy in FISCAL_YEARS:
            total=sum(1 for c,y in panel if y==fy); cov=covered[fy]
            yearlines.append(f"{fy}: panel={total}, anchors={cov}, missing={total-cov}, coverage={cov/total:.6f}")

        print("[6/6] 写出可上传报告……", flush=True)
        lines=[
            "PeerJ 141707 | Label v1 Stage B3b corrected CNINFO annual-report public anchor",
            "NO FINAL LABELS; NO TRAINING; NO PDF DOWNLOAD",
            "Local time: "+datetime.now().astimezone().isoformat(), "Output: "+str(out), "",
            "A. WHY B3 IS SUPERSEDED", "B3_plate_was_blank=true",
            "B3_result_should_not_be_used_as_CNINFO_coverage_evidence=true",
            "corrected_routes=sse+sh;sse+shkcp;szse+sz;szse+szcy", "",
            "B. PANEL PREFIX COUNTS", json.dumps(dict(prefixes),ensure_ascii=False,sort_keys=True), "",
            "C. ROUTE PROBES", json.dumps(probes,ensure_ascii=False,sort_keys=True), "",
            "D. RETRIEVAL", f"raw_announcement_records_seen={raw_seen}",
            f"panel_and_fiscal_year_title_matched_records={matched}",
            "request_sources="+json.dumps(dict(req),ensure_ascii=False,sort_keys=True),
            f"polite_delay_seconds={args.delay}", "concurrency=1", "pdf_downloads=0", "",
            "E. PANEL ANCHOR COVERAGE", f"panel_n={len(panel)}", f"anchor_n={n_cov}",
            f"missing_anchor_n={len(panel)-n_cov}", f"coverage={coverage:.8f}",
            f"normal_t_plus_1_anchor_n={normal}", f"timing_anomaly_n={len(anomalies)}",
            "anchor_rules="+json.dumps(dict(rules),ensure_ascii=False,sort_keys=True),
            "adjunct_path_date_crosscheck="+json.dumps(dict(pathcheck),ensure_ascii=False,sort_keys=True),
            f"bse_or_neeq_panel_rows_not_handled_by_four_routes={prefixes['bse_or_neeq']}",
            f"other_prefix_panel_rows={prefixes['other']}", "",
            "F. YEAR-BY-YEAR COVERAGE", *yearlines, "", "G. DECISION GATE"]
        if coverage>=.95 and normal/max(1,n_cov)>=.99:
            lines += ["public_anchor_source_eligible_for_freeze=true",
                      "Corrected CNINFO routes provide sufficient coverage and timing plausibility.",
                      "Residual missing/BSE/other-prefix rows remain unresolved and are not backfilled from audit dates.",
                      "STATUS: LABEL_V1_STAGE_B3B_PUBLIC_ANCHOR_READY_FOR_FREEZE"]
        else:
            lines += ["public_anchor_source_eligible_for_freeze=false",
                      "Corrected route coverage/timing gate was not met.",
                      "Residual missingness must be diagnosed by code-prefix/route before post-filing inference.",
                      "STATUS: LABEL_V1_STAGE_B3B_STOP_COVERAGE_OR_TIMING_GATE"]
        lines += ["", "source_files_modified=false", "training_or_prediction_run=false",
                  "final_nonnull_labels_written=0", "Upload only label_v1_stageB3b_report.txt."]
        report.write_text("\n".join(lines)+"\n",encoding="utf-8")
        (out/"manifest_v1_stageB3b.json").write_text(json.dumps({
            "created_at":datetime.now().astimezone().isoformat(), "panel_sha256":PANEL_SHA256,
            "coverage":coverage, "anchor_output_sha256":sha256(out/"annual_report_public_anchors_CNINFO_v2.parquet"),
            "final_labels_authorized":False, "training_authorized":False},
            ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        status=[x for x in lines if x.startswith("STATUS:")][0]
        print(status); print("请只上传：",report)
        if args.reveal: reveal(report)

    except Exception as e:
        report.write_text("PeerJ 141707 | Stage B3b FAILED SAFELY\n"
                          f"ERROR_TYPE={type(e).__name__}\nERROR={e}\n"
                          "source_files_modified=false\ntraining_or_prediction_run=false\nfinal_nonnull_labels_written=0\n"
                          "STATUS: LABEL_V1_STAGE_B3B_FAILED_SAFE\n",encoding="utf-8")
        if args.reveal: reveal(report)
        raise

if __name__ == "__main__":
    main()
