#!/usr/bin/env python3
"""
PeerJ 141707 — Label v1 Stage B3d
CNINFO annual-report public-anchor crawl with route inception handling.

Fixes B3c:
- STAR Market did not exist for fiscal years 2010–2018; those route-year
  combinations are skipped instead of treated as malformed responses.
- CNINFO zero-result responses are accepted when totalRecordNum/totalpages
  indicate zero records, even if `announcements` is null/missing.
- Reuses B3c cache where safe to minimize repeat requests.
- Still: no PDF download, no training, no final fraud labels.
- Finder reveal retained.
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

OUT_ROOT = BASE / "label_v1_stageB3d"
# Reuse B3c cache to avoid repeat network requests.
CACHE_ROOT = BASE / "cninfo_public_anchor_cache_v3"

URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
REFERER = "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search"
CATEGORY = "category_ndbg_szsh;"
PAGE_SIZE = 30
BJ = ZoneInfo("Asia/Shanghai")
FISCAL_YEARS = range(2010, 2025)

ROUTES = (
    {"name":"sse_main", "column":"sse",  "plate":"sh",
     "panel_re":r"^60", "exchange_re":r"^(60|68|90)", "min_fy":2010},
    {"name":"sse_star", "column":"sse",  "plate":"shkcp",
     "panel_re":r"^68", "exchange_re":r"^(60|68|90)", "min_fy":2019},
    {"name":"szse_main","column":"szse", "plate":"sz",
     "panel_re":r"^00", "exchange_re":r"^(00|20|30)", "min_fy":2010},
    {"name":"szse_gem", "column":"szse", "plate":"szcy",
     "panel_re":r"^30", "exchange_re":r"^(00|20|30)", "min_fy":2010},
)

FULL_RE = re.compile(r"(?P<year>20\d{2})\s*(?:年\s*)?(?:年度报告|年报)", re.I)
SUMMARY_RE = re.compile(r"摘要|summary", re.I)
ENGLISH_RE = re.compile(r"英文|english", re.I)
NONREPORT_RE = re.compile(
    r"问询|回复|说明|提示|预约|审计报告|鉴证报告|审核报告|社会责任|"
    r"环境社会|ESG|可持续|内部控制|董事会|监事会", re.I)
REVISED_RE = re.compile(r"修订|更正|更新|补充|修正版|修订版", re.I)

def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def load_panel():
    import pyarrow.parquet as pq
    if sha256(PANEL)!=PANEL_SHA256:
        raise ValueError("PANEL_HASH_MISMATCH")
    d=pq.read_table(PANEL,columns=["firm_id","year"],use_threads=False).to_pydict()
    keys={(str(f),int(y)) for f,y in zip(d["firm_id"],d["year"])}
    if len(keys)!=51675:
        raise ValueError(f"PANEL_COUNT_UNEXPECTED:{len(keys)}")
    return keys

def payload(route,page,cal_year):
    return {
        "pageNum":str(page),"pageSize":str(PAGE_SIZE),"tabName":"fulltext",
        "column":route["column"],"stock":"","searchkey":"","secid":"",
        "plate":route["plate"],"category":CATEGORY,"trade":"",
        "seDate":f"{cal_year}-01-01~{cal_year}-12-31",
        "sortName":"code","sortType":"asc","isHLtitle":"true",
    }

def request_json(data, timeout=30, retries=4):
    body=urllib.parse.urlencode(data).encode()
    headers={
        "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153 Safari/537.36",
        "Content-Type":"application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With":"XMLHttpRequest","Origin":"https://www.cninfo.com.cn",
        "Referer":REFERER,"Accept":"application/json, text/javascript, */*; q=0.01",
    }
    last=None
    for i in range(retries):
        try:
            req=urllib.request.Request(URL,data=body,headers=headers,method="POST")
            with urllib.request.urlopen(req,timeout=timeout) as r:
                raw=r.read()
                if r.status!=200: raise RuntimeError(f"HTTP_{r.status}")
                return json.loads(raw.decode("utf-8")),raw
        except Exception as e:
            last=e
            if i+1<retries:
                time.sleep(min(12,1.5*2**i)+random.uniform(.1,.5))
    raise RuntimeError(f"CNINFO_REQUEST_FAILED:{type(last).__name__}:{last}")

def cpath(route,fy,page):
    d=CACHE_ROOT/route["name"]/str(fy)
    d.mkdir(parents=True,exist_ok=True,mode=0o700)
    return d/f"page_{page:04d}.json"

def _zero_result_shape(obj):
    if not isinstance(obj,dict):
        return False
    total = obj.get("totalRecordNum")
    totalpages = obj.get("totalpages", obj.get("totalPages"))
    try:
        if total is not None and int(total)==0:
            return True
    except Exception:
        pass
    try:
        if totalpages is not None and int(totalpages)==0:
            return True
    except Exception:
        pass
    return False

def _normalize_obj(obj):
    if not isinstance(obj,dict):
        raise RuntimeError("CNINFO_RESPONSE_NOT_DICT")
    anns=obj.get("announcements")
    if isinstance(anns,list):
        return obj
    if anns is None and _zero_result_shape(obj):
        obj=dict(obj); obj["announcements"]=[]
        return obj
    raise RuntimeError("CNINFO_RESPONSE_SHAPE_UNEXPECTED")

def fetch(route,fy,page,delay):
    p=cpath(route,fy,page)
    if p.is_file():
        obj=json.loads(p.read_text(encoding="utf-8"))
        try:
            return _normalize_obj(obj),"cache"
        except Exception:
            # An old cached malformed/partial page must not poison the run.
            p.unlink(missing_ok=True)
    obj,raw=request_json(payload(route,page,fy+1))
    obj=_normalize_obj(obj)
    tmp=p.with_suffix(".tmp"); tmp.write_bytes(raw); os.replace(tmp,p)
    time.sleep(delay+random.uniform(0,.2))
    return obj,"network"

def pages(obj):
    obj=_normalize_obj(obj)
    for k in ("totalpages","totalPages"):
        try:
            n=int(obj.get(k))
            if n>=0:return n
        except Exception:pass
    try:
        n=int(obj.get("totalRecordNum") or 0)
        return math.ceil(n/PAGE_SIZE) if n>0 else 0
    except Exception:
        pass
    return 1 if obj.get("announcements") else 0

def clean(s):
    return re.sub(r"\s+","",re.sub(r"</?em>","",str(s or ""),flags=re.I))

def classify(title):
    t=clean(title); m=FULL_RE.search(t)
    if not m:return None,"not_annual"
    fy=int(m.group("year"))
    if SUMMARY_RE.search(t):return fy,"summary"
    if ENGLISH_RE.search(t):return fy,"english"
    if NONREPORT_RE.search(t):return fy,"related_nonreport"
    if REVISED_RE.search(t):return fy,"full_revised"
    return fy,"full_original"

def ann_date(ms):
    try:return datetime.fromtimestamp(int(ms)/1000,tz=timezone.utc).astimezone(BJ).date()
    except Exception:return None

def url_date(u):
    m=re.search(r"(?:^|/)finalpage/(\d{4})-(\d{2})-(\d{2})/",str(u or ""))
    if not m:return None
    try:return date(int(m.group(1)),int(m.group(2)),int(m.group(3)))
    except ValueError:return None

def choose(cands):
    a=[x for x in cands if x["class"]=="full_original"]
    if a:return min(a,key=lambda z:z["date"]),"full_original_earliest"
    a=[x for x in cands if x["class"]=="full_revised"]
    if a:return min(a,key=lambda z:z["date"]),"full_revised_fallback"
    return None,"missing"

def probe(route,delay):
    fy=max(2023,route["min_fy"])
    obj,src=fetch(route,fy,1,delay)
    anns=[x for x in (obj.get("announcements") or [])[:30] if isinstance(x,dict)]
    codes=[str(x.get("secCode") or "") for x in anns if x.get("secCode")]
    exchange_purity=sum(bool(re.search(route["exchange_re"],c)) for c in codes)/len(codes) if codes else 0
    panel_prefix_share=sum(bool(re.search(route["panel_re"],c)) for c in codes)/len(codes) if codes else 0
    prefix2=Counter(c[:2] for c in codes if len(c)>=2)
    fields=sum(bool(x.get("secCode")) and bool(x.get("announcementTitle")) and x.get("announcementTime") is not None for x in anns)
    return {
        "ok":bool(anns) and fields>=max(1,len(anns)//2) and exchange_purity>=.95,
        "source":src,"pages":pages(obj),"records":len(obj.get("announcements") or []),
        "exchange_purity":exchange_purity,"panel_A_prefix_share":panel_prefix_share,
        "sample_prefix2":dict(prefix2),
    }

def write_parquet(path,rows):
    import pyarrow as pa, pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows),path,compression="zstd")

def reveal(p):
    if sys.platform=="darwin":
        subprocess.run(["open","-R",str(p)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage-b3d",action="store_true")
    ap.add_argument("--delay",type=float,default=1.2)
    ap.add_argument("--reveal",action="store_true")
    a=ap.parse_args()
    if not a.stage_b3d:ap.error("--stage-b3d required")
    if a.delay<.8:ap.error("--delay must be >= 0.8")

    os.umask(0o077)
    OUT_ROOT.mkdir(parents=True,exist_ok=True,mode=0o700)
    CACHE_ROOT.mkdir(parents=True,exist_ok=True,mode=0o700)
    out=OUT_ROOT/("stageB3d_"+datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8])
    out.mkdir(mode=0o700)
    report=out/"label_v1_stageB3d_report.txt"

    try:
        print("[1/6] 核验固定51,675面板……",flush=True)
        panel=load_panel()
        panel_routes=Counter()
        for code,fy in panel:
            for r in ROUTES:
                if fy>=r["min_fy"] and re.search(r["panel_re"],code):
                    panel_routes[r["name"]]+=1; break

        print("[2/6] 探测四条CNINFO路由……",flush=True)
        probes={}
        for r in ROUTES:
            probes[r["name"]]=probe(r,a.delay)
            p=probes[r["name"]]
            print(f"      {r['name']}: ok={p['ok']} pages={p['pages']} exchange={p['exchange_purity']:.1%} prefixes={p['sample_prefix2']}",flush=True)
        if not all(x["ok"] for x in probes.values()):
            lines=[
                "PeerJ 141707 | Label v1 Stage B3d route probe",
                "NO FINAL LABELS; NO TRAINING; NO PDF DOWNLOAD",
                "panel_route_counts="+json.dumps(dict(panel_routes),ensure_ascii=False,sort_keys=True),
                "probes="+json.dumps(probes,ensure_ascii=False,sort_keys=True),
                "full_crawl_started=false",
                "STATUS: LABEL_V1_STAGE_B3D_STOP_ROUTE_PROBE_FAILED",
                "source_files_modified=false","training_or_prediction_run=false","final_nonnull_labels_written=0",
            ]
            report.write_text("\n".join(lines)+"\n",encoding="utf-8")
            print(lines[-4]); print("请上传：",report)
            if a.reveal:reveal(report)
            return

        print("[3/6] 单线程、可续传抓取2010–2024财年公开年报公告元数据……",flush=True)
        bykey=defaultdict(list); req=Counter(); raw=0; matched=0; route_year=[]
        skipped_route_years=[]

        for fy in FISCAL_YEARS:
            for r in ROUTES:
                if fy < r["min_fy"]:
                    skipped_route_years.append({"fiscal_year":fy,"route":r["name"],"reason":"pre_route_inception"})
                    continue

                obj,src=fetch(r,fy,1,a.delay); req[src]+=1
                npage=pages(obj)
                if npage<0 or npage>1000:
                    raise RuntimeError(f"BAD_PAGE_COUNT:{r['name']}:{fy}:{npage}")

                rr=kk=0
                if npage==0:
                    route_year.append({"fiscal_year":fy,"route":r["name"],"pages":0,"raw":0,"matched":0})
                    print(f"      FY{fy} {r['name']}: pages=0 (zero-result)",flush=True)
                    continue

                for pg in range(1,npage+1):
                    if pg==1:
                        cur=obj
                    else:
                        cur,src2=fetch(r,fy,pg,a.delay); req[src2]+=1
                    for x in cur.get("announcements") or []:
                        if not isinstance(x,dict):continue
                        raw+=1;rr+=1
                        code=str(x.get("secCode") or "").strip()
                        if not re.fullmatch(r"\d{6}",code):continue
                        if not re.search(r["panel_re"],code):continue
                        key=(code,fy)
                        if key not in panel:continue
                        tfy,cl=classify(x.get("announcementTitle"))
                        if tfy!=fy:continue
                        d=ann_date(x.get("announcementTime"))
                        if not d:continue
                        ud=url_date(x.get("adjunctUrl"))
                        matched+=1;kk+=1
                        bykey[key].append({
                            "date":d,"class":cl,"title":clean(x.get("announcementTitle")),
                            "id":str(x.get("announcementId") or ""),"url":str(x.get("adjunctUrl") or ""),
                            "route":r["name"],"url_date":ud,
                        })
                route_year.append({"fiscal_year":fy,"route":r["name"],"pages":npage,"raw":rr,"matched":kk})
                print(f"      FY{fy} {r['name']}: pages={npage}, raw={rr}, matched={kk}",flush=True)

        print("[4/6] 生成最早完整年报公开日期……",flush=True)
        rows=[]; cover=Counter(); rule=Counter(); cross=Counter(); anomalies=[]
        for code,fy in sorted(panel,key=lambda z:(z[1],z[0])):
            z,ru=choose(bykey.get((code,fy),[]))
            if not z:
                rows.append({"company_code":code,"fiscal_year":fy,"annual_report_public_anchor":None,"anchor_rule":"missing"})
                continue
            d=z["date"];cover[fy]+=1;rule[ru]+=1
            if z["url_date"]==d:cross["match"]+=1
            elif z["url_date"] is None:cross["no_url_date"]+=1
            else:cross["mismatch"]+=1
            flag="normal_t_plus_1"
            if d<=date(fy,12,31):flag="on_or_before_fiscal_year_end"
            elif d>date(fy+1,12,31):flag="after_t_plus_1_year_end"
            if flag!="normal_t_plus_1":
                anomalies.append({"company_code":"'"+code,"fiscal_year":fy,"date":d.isoformat(),"flag":flag,"title":z["title"],"id":z["id"]})
            rows.append({
                "company_code":code,"fiscal_year":fy,"annual_report_public_anchor":d.isoformat(),
                "anchor_rule":ru,"timing_flag":flag,"announcement_id":z["id"],
                "announcement_title":z["title"],"adjunct_url":z["url"],"route":z["route"],
                "source":"CNINFO announcement metadata; route-inception aware; panel-filtered",
            })

        write_parquet(out/"annual_report_public_anchors_CNINFO_v4.parquet",rows)

        with (out/"anchor_anomalies_STAGEB3D_LOCAL_ONLY.csv").open("w",encoding="utf-8",newline="") as f:
            fields=["company_code","fiscal_year","date","flag","title","id"]
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(anomalies)

        with (out/"route_year_summary.csv").open("w",encoding="utf-8",newline="") as f:
            fields=["fiscal_year","route","pages","raw","matched"]
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(route_year)

        print("[5/6] 计算全样本和逐年覆盖……",flush=True)
        n=sum(cover.values()); cov=n/len(panel)
        normal=sum(1 for r in rows if r.get("annual_report_public_anchor") and r.get("timing_flag")=="normal_t_plus_1")
        yearlines=[]
        for fy in FISCAL_YEARS:
            total=sum(1 for c,y in panel if y==fy); cc=cover[fy]
            yearlines.append(f"{fy}: panel={total}, anchors={cc}, missing={total-cc}, coverage={cc/total:.6f}")

        print("[6/6] 写出报告……",flush=True)
        lines=[
            "PeerJ 141707 | Label v1 Stage B3d CNINFO public annual-report anchor",
            "NO FINAL LABELS; NO TRAINING; NO PDF DOWNLOAD",
            "Local time: "+datetime.now().astimezone().isoformat(),
            "Output: "+str(out),"",
            "A. FIX FROM B3C",
            "sse_star_min_fiscal_year=2019",
            "zero_result_response_is_valid_when_total_is_zero=true",
            "B3c_cache_reused=true","",
            "B. PANEL ROUTE COUNTS",
            json.dumps(dict(panel_routes),ensure_ascii=False,sort_keys=True),"",
            "C. ROUTE PROBES",
            json.dumps(probes,ensure_ascii=False,sort_keys=True),"",
            "D. RETRIEVAL",
            f"raw_records_seen={raw}",f"panel_title_matched_records={matched}",
            "request_sources="+json.dumps(dict(req),ensure_ascii=False,sort_keys=True),
            f"skipped_pre_inception_route_years={len(skipped_route_years)}",
            f"polite_delay_seconds={a.delay}","concurrency=1","pdf_downloads=0","",
            "E. PANEL COVERAGE",
            f"panel_n={len(panel)}",f"anchor_n={n}",f"missing_n={len(panel)-n}",
            f"coverage={cov:.8f}",f"normal_t_plus_1_n={normal}",
            "anchor_rules="+json.dumps(dict(rule),ensure_ascii=False,sort_keys=True),
            "url_date_crosscheck="+json.dumps(dict(cross),ensure_ascii=False,sort_keys=True),
            f"timing_anomaly_n={len(anomalies)}","",
            "F. YEAR-BY-YEAR COVERAGE",*yearlines,"",
            "G. DECISION",
        ]
        if cov>=.95 and normal/max(1,n)>=.99:
            lines += [
                "public_anchor_source_eligible_for_freeze=true",
                "Residual missing/anomalous firm-years remain unresolved and will not be backfilled from audit dates.",
                "STATUS: LABEL_V1_STAGE_B3D_PUBLIC_ANCHOR_READY_FOR_FREEZE",
            ]
        else:
            lines += [
                "public_anchor_source_eligible_for_freeze=false",
                "Coverage/timing gate not met; residual missingness needs targeted diagnosis only.",
                "STATUS: LABEL_V1_STAGE_B3D_STOP_COVERAGE_OR_TIMING_GATE",
            ]
        lines += ["","source_files_modified=false","training_or_prediction_run=false","final_nonnull_labels_written=0","Upload only label_v1_stageB3d_report.txt."]
        report.write_text("\n".join(lines)+"\n",encoding="utf-8")

        (out/"manifest_v1_stageB3d.json").write_text(json.dumps({
            "created_at":datetime.now().astimezone().isoformat(),
            "panel_sha256":PANEL_SHA256,"coverage":cov,
            "anchor_sha256":sha256(out/"annual_report_public_anchors_CNINFO_v4.parquet"),
            "final_labels_authorized":False,"training_authorized":False,
        },ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")

        status=[x for x in lines if x.startswith("STATUS:")][0]
        print(status);print("请只上传：",report)
        if a.reveal:reveal(report)

    except Exception as e:
        report.write_text(
            "PeerJ 141707 | Stage B3d FAILED SAFELY\n"
            f"ERROR_TYPE={type(e).__name__}\nERROR={e}\n"
            "source_files_modified=false\ntraining_or_prediction_run=false\nfinal_nonnull_labels_written=0\n"
            "STATUS: LABEL_V1_STAGE_B3D_FAILED_SAFE\n",
            encoding="utf-8")
        if a.reveal:reveal(report)
        raise

if __name__=="__main__":
    main()
