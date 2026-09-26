#!/usr/bin/env python3
"""
PeerJ 141707 — Label v1 Stage B2
Search for a TRUE annual-report public disclosure/filing date source.

Important:
- FIN_Audit.Annodt is explicitly excluded because Stage B1 descriptor defines it as “审计日期”.
- Read-only inputs; writes only under ~/peerj_141707_audit/label_v1_stageB2/
- No network, no training, no final 0/1 labels.
- --reveal opens Finder and selects the report on macOS.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, os, re, subprocess, sys, time, uuid, zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

BASE = Path.home() / "peerj_141707_audit"
PROJECT = Path(os.environ.get("PEERJ_SOURCE_PROJECT_ROOT", "private_inputs/project"))
CSMAR = Path(os.environ.get("PEERJ_CSMAR_ROOT", "private_inputs/csmar"))
PANEL = PROJECT / "data/processed/kg/node_features_v1_1.parquet"
PANEL_SHA256 = "adc00129f06ffb0995691c6d1ba98f740c53561c1e0cc4e91d927b84eb36beb2"
REJECTED = Path(os.environ.get("PEERJ_AUDIT_SOURCE_XLSX", "private_inputs/csmar/audit/FIN_Audit.xlsx")).resolve()
START = time.monotonic()
MAX_SECONDS = 1800

PUBLIC = re.compile(
    r"公告日期|公告时间|公布日期|发布日期|披露日期|披露时间|首次披露|实际披露|"
    r"年报披露|年度报告披露|年报公告|年度报告公告|"
    r"announcement.?date|announcement.?time|publish.?date|publication.?date|"
    r"disclosure.?date|filing.?date|actual.?publish|announcementtime|pubdate", re.I)
ANNUAL = re.compile(r"年度报告|年报|annual.?report|报告期|会计期间|Accper", re.I)
AUDIT_ONLY = re.compile(r"审计日期|审计报告日|审计报告日期|签字日期|签署日期|audit.?report.?date", re.I)

CODE_ALIASES = {"firm_id","company_code","stock_code","stkcd","stockcode","symbol","secu_code","secucode","证券代码","股票代码","公司代码","代码"}
YEAR_ALIASES = {"year","fiscal_year","report_year","报告年度","会计年度","年度"}
PERIOD_ALIASES = {"accper","report_period","report_date","end_date","enddate","截止日期","报告期","会计期间","报表日期","报告期末"}
DATE_ALIASES = {"announcement_date","announce_date","announcement_time","announcetime","publish_date","publication_date","filing_date","disclosure_date","actual_publish_time","actual_publish_date","pubdate","pub_date","公告日期","公告时间","公布日期","发布日期","披露日期","披露时间","首次披露日期","实际披露日期","年报披露日期","年度报告披露日期","报告披露日期"}
TYPE_ALIASES = {"report_type","reporttype","typrep","type","公告类型","报告类型","报表类型","公告类别"}

def tick():
    if time.monotonic() - START > MAX_SECONDS:
        raise RuntimeError("TIME_LIMIT_1800_SECONDS")

def sha256(p: Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def norm(x):
    return re.sub(r"[\s_\-./（）()【】\[\]]+","",("" if x is None else str(x).strip())).lower()

def find_alias(headers, aliases):
    aa={norm(x) for x in aliases}
    for h in headers:
        if norm(h) in aa: return h
    return None

def parse_code(x):
    if x is None:return None
    s=str(x).strip()
    if re.fullmatch(r"\d+(?:\.0+)?",s):
        try:return f"{int(float(s)):06d}"[-6:]
        except Exception:pass
    m=re.search(r"(?<!\d)(\d{6})(?!\d)",s)
    return m.group(1) if m else None

def excel_date(x):
    try:v=float(x)
    except Exception:return None
    if 20000<=v<=80000:return date(1899,12,30)+timedelta(days=int(v))
    return None

def parse_date(x):
    if x is None:return None
    if isinstance(x,datetime):return x.date()
    if isinstance(x,date):return x
    d=excel_date(x)
    if d:return d
    s=str(x).strip().replace("年","-").replace("月","-").replace("日","").replace("/", "-").replace(".", "-")
    m=re.search(r"((?:19|20)\d{2})-?(\d{1,2})-?(\d{1,2})",s)
    if not m:return None
    try:return date(int(m.group(1)),int(m.group(2)),int(m.group(3)))
    except ValueError:return None

def parse_year(x):
    if x is None:return None
    if isinstance(x,(int,float)) and not isinstance(x,bool):
        y=int(x); return y if 1990<=y<=2030 else None
    m=re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)",str(x))
    return int(m.group(1)) if m else None

def cell_col(ref):
    m=re.match(r"([A-Z]+)",ref or "")
    if not m:return None
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
    except KeyError: pass
    return out

def sheet_targets(z):
    nsm="{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    nsr="{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    nsp="{http://schemas.openxmlformats.org/package/2006/relationships}"
    wb=ET.parse(z.open("xl/workbook.xml")).getroot()
    rr=ET.parse(z.open("xl/_rels/workbook.xml.rels")).getroot()
    rel={r.attrib["Id"]:r.attrib["Target"] for r in rr.findall(nsp+"Relationship")}
    ans=[]
    for sh in wb.find(nsm+"sheets"):
        t=rel.get(sh.attrib.get(nsr+"id"))
        if t:
            if not t.startswith("xl/"):t="xl/"+t.lstrip("/")
            ans.append((sh.attrib.get("name",""),t))
    return ans

def xlsx_headers(path: Path):
    ns="{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    out=[]
    with zipfile.ZipFile(path) as z:
        ss=shared_strings(z)
        for sheet,target in sheet_targets(z)[:4]:
            with z.open(target) as f:
                for ev,elem in ET.iterparse(f,events=("end",)):
                    if elem.tag!=ns+"row":continue
                    vals={}
                    for c in elem.findall(ns+"c"):
                        i=cell_col(c.attrib.get("r"))
                        if i is None:continue
                        typ=c.attrib.get("t"); v=c.find(ns+"v")
                        val=None if v is None else v.text
                        if typ=="s" and val is not None:
                            try:val=ss[int(val)]
                            except Exception:pass
                        elif typ=="inlineStr":
                            isel=c.find(ns+"is")
                            if isel is not None: val="".join((t.text or "") for t in isel.iter(ns+"t"))
                        vals[i]=val
                    if vals:
                        m=max(vals); out.append((f"xlsx:{sheet}",[vals.get(i) for i in range(m+1)]))
                    elem.clear(); break
    return out

def csv_header(path: Path):
    for enc in ("utf-8-sig","utf-8","gb18030"):
        try:
            s=path.open("r",encoding=enc,errors="strict").read(16384)
            if not s:return None
            delim="\t" if s.count("\t")>s.count(",") else ","
            return next(csv.reader(io.StringIO(s),delimiter=delim)),enc,delim
        except Exception:pass
    return None

def file_headers(path: Path):
    try:
        ext=path.suffix.lower()
        if ext==".parquet":
            import pyarrow.parquet as pq
            return [("parquet",pq.ParquetFile(path).schema_arrow.names)]
        if ext in (".csv",".tsv"):
            h=csv_header(path)
            return [(f"csv:{h[1]}:{repr(h[2])}",h[0])] if h else []
        if ext==".xlsx":
            return xlsx_headers(path)
        if ext in (".jsonl",".ndjson"):
            with path.open("r",encoding="utf-8",errors="ignore") as f:
                for line in f:
                    if line.strip():
                        obj=json.loads(line)
                        if isinstance(obj,dict):return [("jsonl",list(obj))]
        if ext==".json" and path.stat().st_size<50*1024*1024:
            obj=json.loads(path.read_text(encoding="utf-8",errors="ignore"))
            if isinstance(obj,list) and obj and isinstance(obj[0],dict):return [("json",list(obj[0]))]
            if isinstance(obj,dict):
                for k in ("data","items","records","results","announcements"):
                    v=obj.get(k)
                    if isinstance(v,list) and v and isinstance(v[0],dict):return [(f"json:{k}",list(v[0]))]
    except Exception:pass
    return []

def make_spec(path,fmt,headers,context):
    if path.resolve()==REJECTED:return None
    code=find_alias(headers,CODE_ALIASES); year=find_alias(headers,YEAR_ALIASES)
    period=find_alias(headers,PERIOD_ALIASES); dcol=find_alias(headers,DATE_ALIASES)
    typ=find_alias(headers,TYPE_ALIASES)
    if not code or not dcol or not (year or period):return None
    sem=context+" "+" ".join(map(str,headers))
    if AUDIT_ONLY.search(sem) and not PUBLIC.search(sem):return None
    return {"path":str(path),"format":fmt,"code":code,"year":year,"period":period,"date":dcol,"type":typ,"semantic":context[:1200]}

def load_panel():
    import pyarrow.parquet as pq
    if sha256(PANEL)!=PANEL_SHA256:raise ValueError("PANEL_HASH_MISMATCH")
    t=pq.read_table(PANEL,columns=["firm_id","year"],use_threads=False).to_pydict()
    keys={(str(f),int(y)) for f,y in zip(t["firm_id"],t["year"])}
    if len(keys)!=51675:raise ValueError("PANEL_COUNT_UNEXPECTED")
    return keys

def descriptor_specs():
    ans=[]; inspected=0
    for p in CSMAR.rglob("*.txt"):
        tick();inspected+=1
        if inspected>6000:break
        try:
            if p.stat().st_size>10*1024*1024:continue
            txt=None
            for enc in ("utf-8","gb18030","gbk"):
                try:txt=p.read_text(encoding=enc);break
                except Exception:pass
            if not txt or not PUBLIC.search(txt) or not ANNUAL.search(txt):continue
        except Exception:continue
        snippets=[]
        ls=txt.splitlines()
        for i,line in enumerate(ls):
            if PUBLIC.search(line):
                snippets.append(" | ".join(x.strip() for x in ls[max(0,i-2):min(len(ls),i+3)] if x.strip())[:1000])
                if len(snippets)>=4:break
        context=p.name+" :: "+" || ".join(snippets)
        for q in p.parent.iterdir():
            if q.is_file() and q.suffix.lower() in {".xlsx",".csv",".tsv",".parquet",".json",".jsonl",".ndjson"}:
                for fmt,h in file_headers(q):
                    s=make_spec(q,fmt,h,context)
                    if s:ans.append(s)
    return ans

def project_specs():
    ans=[]; seen=set()
    for root in (PROJECT/"data",PROJECT/"scripts"):
        if not root.is_dir():continue
        count=0
        for base,dirs,files in os.walk(root):
            tick()
            dirs[:]=[d for d in dirs if d not in {".venv","venv","__pycache__","models","checkpoints","pdfs","embeddings","chunks"}]
            if len(Path(base).relative_to(root).parts)>5:
                dirs[:]=[];continue
            for fn in files:
                count+=1
                if count>20000:dirs[:]=[];break
                p=Path(base)/fn
                if p.suffix.lower() not in {".xlsx",".csv",".tsv",".parquet",".json",".jsonl",".ndjson"}:continue
                if p.stat().st_size>300*1024*1024:continue
                if not re.search(r"annual|report|announce|publish|disclos|meta|manifest|index|年报|公告|披露",str(p),re.I):continue
                if p.resolve() in seen or p.resolve()==REJECTED:continue
                seen.add(p.resolve())
                for fmt,h in file_headers(p):
                    s=make_spec(p,fmt,h,str(p))
                    if s:ans.append(s)
    return ans

def iter_records(spec):
    path=Path(spec["path"]); ext=path.suffix.lower()
    cols=[x for x in (spec["code"],spec["year"],spec["period"],spec["date"],spec["type"]) if x]
    if ext==".parquet":
        import pyarrow.parquet as pq
        for r in pq.read_table(path,columns=cols,use_threads=False).to_pylist():yield r
    elif ext in (".csv",".tsv"):
        h,enc,delim=csv_header(path)
        with path.open("r",encoding=enc,errors="ignore",newline="") as f:
            for r in csv.DictReader(f,delimiter=delim):yield {k:r.get(k) for k in cols}
    elif ext==".xlsx":
        sheet=spec["format"].split(":",1)[1]
        # simple full sheet iterator
        ns="{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        with zipfile.ZipFile(path) as z:
            ss=shared_strings(z); target=dict(sheet_targets(z))[sheet]
            header=None; idx=None
            with z.open(target) as f:
                for ev,elem in ET.iterparse(f,events=("end",)):
                    if elem.tag!=ns+"row":continue
                    vals={}
                    for c in elem.findall(ns+"c"):
                        i=cell_col(c.attrib.get("r")); typ=c.attrib.get("t"); v=c.find(ns+"v")
                        val=None if v is None else v.text
                        if typ=="s" and val is not None:
                            try:val=ss[int(val)]
                            except Exception:pass
                        vals[i]=val
                    elem.clear()
                    if not vals:continue
                    m=max(vals); row=[vals.get(i) for i in range(m+1)]
                    if header is None:
                        header=[("" if x is None else str(x).strip()) for x in row]; idx={h:i for i,h in enumerate(header)};continue
                    yield {k:(row[idx[k]] if k in idx and idx[k]<len(row) else None) for k in cols}

def score_spec(spec,panel):
    mp={}; datesets=defaultdict(set); seen=0
    for r in iter_records(spec):
        tick();seen+=1
        if seen>2_000_000:break
        code=parse_code(r.get(spec["code"]))
        y=parse_year(r.get(spec["year"])) if spec["year"] else None
        per=r.get(spec["period"]) if spec["period"] else None
        if y is None:y=parse_year(per)
        d=parse_date(r.get(spec["date"]))
        if not code or y is None or d is None or not 2010<=y<=2024:continue
        pd=parse_date(per) if per is not None else None
        if pd and not (pd.month==12 and pd.day==31):continue
        k=(code,y)
        if k not in panel:continue
        datesets[k].add(d)
        if k not in mp or d<mp[k]:mp[k]=d
    n=len(mp)
    pre=sum(d<=date(y,12,31) for (c,y),d in mp.items())
    t1=sum(date(y+1,1,1)<=d<=date(y+1,12,31) for (c,y),d in mp.items())
    june=sum(date(y+1,1,1)<=d<=date(y+1,6,30) for (c,y),d in mp.items())
    multi=sum(len(v)>1 for v in datesets.values())
    pub=bool(PUBLIC.search(spec["semantic"]+" "+spec["date"]))
    ann=bool(ANNUAL.search(spec["semantic"]+" "+str(spec["period"] or "")))
    return {
        "path":spec["path"],"format":spec["format"],"date_col":spec["date"],
        "code_col":spec["code"],"year_col":spec["year"],"period_col":spec["period"],
        "semantic_public":pub,"semantic_annual":ann,
        "anchor_n":n,"coverage":n/len(panel) if panel else 0,
        "t_plus_1_rate":t1/n if n else 0,
        "jan_jun_rate":june/n if n else 0,
        "pre_fiscal_end":pre,"multiple_dates":multi,"rows_seen":seen,
    }

def reveal(p):
    if sys.platform=="darwin":
        subprocess.run(["open","-R",str(p)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage-b2",action="store_true")
    ap.add_argument("--reveal",action="store_true")
    args=ap.parse_args()
    if not args.stage_b2:ap.error("--stage-b2 required")
    os.umask(0o077)
    root=BASE/"label_v1_stageB2";root.mkdir(mode=0o700,exist_ok=True)
    out=root/("stageB2_"+datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8]);out.mkdir(mode=0o700)
    report=out/"label_v1_stageB2_report.txt"

    try:
        print("[1/5] 核验固定面板；明确排除 FIN_Audit.Annodt（审计日期）……",flush=True)
        panel=load_panel()
        print("[2/5] 搜索 CSMAR 字段说明中的真正公告/披露日期源……",flush=True)
        specs=descriptor_specs()
        print("[3/5] 搜索本地年报下载/manifest/metadata 表……",flush=True)
        specs+=project_specs()
        uniq={}
        for s in specs:
            key=(str(Path(s["path"]).resolve()),s["format"],s["date"],s["code"],s["year"],s["period"])
            uniq[key]=s
        specs=list(uniq.values())
        print(f"      candidates={len(specs)}",flush=True)
        print("[4/5] 计算候选覆盖率与时间合理性……",flush=True)
        scored=[]
        for i,s in enumerate(specs,1):
            try:
                r=score_spec(s,panel);scored.append(r)
                print(f"      {i}/{len(specs)} coverage={r['coverage']:.3%} t+1={r['t_plus_1_rate']:.3%} pre={r['pre_fiscal_end']} :: {Path(r['path']).name}",flush=True)
            except Exception as e:
                scored.append({"path":s["path"],"format":s["format"],"error":f"{type(e).__name__}:{e}","coverage":0,"t_plus_1_rate":0,"pre_fiscal_end":999999})
        scored.sort(key=lambda r:(r.get("semantic_public",False),r.get("semantic_annual",False),r.get("coverage",0),r.get("t_plus_1_rate",0)),reverse=True)
        with (out/"true_anchor_candidates_STAGEB2_LOCAL_ONLY.csv").open("w",encoding="utf-8",newline="") as f:
            fields=sorted({k for r in scored for k in r})
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(scored)

        selected=None
        for r in scored:
            if (r.get("semantic_public") and r.get("semantic_annual") and r.get("coverage",0)>=0.95 and
                r.get("t_plus_1_rate",0)>=0.98 and r.get("pre_fiscal_end",999999)<=5):
                selected=r;break

        print("[5/5] 写出决策报告……",flush=True)
        lines=[
            "PeerJ 141707 | Label v1 Stage B2 true public annual-report anchor recovery",
            "NO FINAL LABELS; NO TRAINING",
            "Local time: "+datetime.now().astimezone().isoformat(),
            "Output: "+str(out),"",
            "A. STAGE-B1 FIXED DECISION",
            "FIN_Audit_Annodt_rejected=true",
            'reason=CSMAR descriptor defines Annodt as "审计日期"; it is not authorized as public annual-report filing/announcement date.',
            "",
            "B. SEARCH SUMMARY",
            f"semantic_schema_candidates={len(specs)}",
            f"scored_candidates={len(scored)}",
        ]
        for i,r in enumerate(scored[:10],1):
            lines += [
                f"candidate_{i}_path={r.get('path')}",
                f"candidate_{i}_format={r.get('format')}",
                f"candidate_{i}_date_col={r.get('date_col')}",
                f"candidate_{i}_coverage={float(r.get('coverage',0)):.8f}",
                f"candidate_{i}_t_plus_1_rate={float(r.get('t_plus_1_rate',0)):.8f}",
                f"candidate_{i}_pre_fiscal_end={r.get('pre_fiscal_end','NA')}",
                f"candidate_{i}_semantic_public={r.get('semantic_public','NA')}",
                f"candidate_{i}_semantic_annual={r.get('semantic_annual','NA')}",
            ]
        if selected:
            lines += ["","C. DECISION","true_public_anchor_selected=true",
                      "selected_path="+selected["path"],
                      "selected_format="+selected["format"],
                      "selected_date_col="+selected["date_col"],
                      f"selected_coverage={selected['coverage']:.8f}",
                      f"selected_t_plus_1_rate={selected['t_plus_1_rate']:.8f}",
                      f"selected_pre_fiscal_end={selected['pre_fiscal_end']}",
                      "STATUS: LABEL_V1_STAGE_B2_TRUE_PUBLIC_ANCHOR_FOUND"]
        else:
            lines += ["","C. DECISION","true_public_anchor_selected=false",
                      "No local source met explicit public-disclosure semantics + annual-report context + coverage/timing gates.",
                      "Next priority: local downloader/API cache with announcementTime/publish/disclosure date; otherwise obtain official exchange/CNINFO announcement metadata.",
                      "STATUS: LABEL_V1_STAGE_B2_STOP_TRUE_PUBLIC_ANCHOR_NOT_FOUND"]
        lines += ["","source_files_modified=false","training_or_prediction_run=false","final_nonnull_labels_written=0",
                  "Upload only label_v1_stageB2_report.txt."]
        report.write_text("\n".join(lines)+"\n",encoding="utf-8")
        print([x for x in lines if x.startswith("STATUS:")][0])
        print("请只上传：",report)
        if args.reveal:reveal(report)
    except Exception as e:
        report.write_text(f"PeerJ 141707 | Stage B2 FAILED SAFELY\nERROR_TYPE={type(e).__name__}\nERROR={e}\nsource_files_modified=false\ntraining_or_prediction_run=false\nfinal_nonnull_labels_written=0\nSTATUS: LABEL_V1_STAGE_B2_FAILED_SAFE\n",encoding="utf-8")
        if args.reveal:reveal(report)
        raise

if __name__=="__main__":
    main()
