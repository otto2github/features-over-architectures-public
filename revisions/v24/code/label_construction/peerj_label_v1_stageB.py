#!/usr/bin/env python3
"""PeerJ 141707 — Label v1 Stage B: annual-report filing anchors + qualifying sanction public dates.

Purpose
-------
Consume a completed Label v1 Stage A output and build the post-filing timing layer.

This stage:
  1) verifies the Stage A manifest and hashes;
  2) discovers LOCAL annual-report filing/announcement-date metadata sources;
  3) scores candidate anchor sources against the fixed 51,675 firm-year panel;
  4) if one source is sufficiently well supported, builds annual-report filing anchors;
  5) verifies the documented meaning of CSMAR DeclareDate / DisposalDate;
  6) attaches earliest qualifying public-sanction dates to Stage-A candidate evidence;
  7) classifies timing as post-filing / pre-or-same-day / unresolved;
  8) summarizes follow-up maturity to the fixed 2026-05-08 cutoff.

It DOES NOT create final fraud labels and DOES NOT train models.
It DOES NOT overwrite any source or Stage A output.
No network access is used.

All outputs are written under:
  ~/peerj_141707_audit/label_v1_stageB/

On macOS, --reveal opens Finder and selects the final report after successful completion.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, date, timedelta
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import uuid
import zipfile
import xml.etree.ElementTree as ET

VERSION = "label_v1_stageB_1.0"
BASE = Path.home() / "peerj_141707_audit"
PROJECT = Path(os.environ.get("PEERJ_SOURCE_PROJECT_ROOT", "private_inputs/project"))
CSMAR_ROOT = Path(os.environ.get("PEERJ_CSMAR_ROOT", "private_inputs/csmar"))
RAW_VIOL_DIR = Path(os.environ.get("PEERJ_VIOLATION_SOURCE_DIR", "private_inputs/csmar/violation"))
PANEL = PROJECT / "data/processed/kg/node_features_v1_1.parquet"
DESCRIPTOR = RAW_VIOL_DIR / "STK_Violation_Main[DES][xlsx].txt"

PANEL_SHA256 = "adc00129f06ffb0995691c6d1ba98f740c53561c1e0cc4e91d927b84eb36beb2"
DESCRIPTOR_SHA256 = "14c4cd27901ee9e4ce629c869b529c8d7ae831df8157dc76d30733bebea55e88"
CUTOFF = date(2026, 5, 8)

START = time.monotonic()
MAX_SECONDS = 1800

CODE_ALIASES = {
    "firm_id", "company_code", "stock_code", "stkcd", "stockcode", "symbol",
    "secu_code", "secucode", "证券代码", "股票代码", "公司代码", "代码"
}
YEAR_ALIASES = {
    "year", "fiscal_year", "report_year", "报告年度", "会计年度", "年度"
}
PERIOD_ALIASES = {
    "accper", "report_period", "report_date", "end_date", "enddate",
    "截止日期", "报告期", "会计期间", "报表日期", "报告期末"
}
ANNOUNCE_ALIASES = {
    "annodt", "announce_date", "announcement_date", "publish_date",
    "publication_date", "filing_date", "disclosure_date", "actual_publish_time",
    "公告日期", "披露日期", "发布日期", "首次披露日期", "报告披露日期",
    "公告时间", "披露时间", "publish_time", "announcement_time"
}
TYPE_ALIASES = {
    "report_type", "reporttype", "typrep", "type", "公告类型", "报告类型", "报表类型"
}

LIKELY_NAME = re.compile(
    r"annual|report|filing|announce|publish|disclos|manifest|meta|index|"
    r"年报|年度报告|公告|披露|审计意见|audit",
    re.I
)

def tick():
    if time.monotonic() - START > MAX_SECONDS:
        raise RuntimeError("TIME_LIMIT_1800_SECONDS")

def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def checked_file(path: Path, max_bytes=1024 * 1024 * 1024):
    if not path.is_absolute():
        raise ValueError("ABSOLUTE_PATH_REQUIRED")
    for p in (path, *path.parents):
        if p.is_symlink():
            raise ValueError(f"SYMLINK_NOT_ALLOWED: {path}")
    st = path.stat()
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"NOT_REGULAR_FILE: {path}")
    if st.st_size > max_bytes:
        raise ValueError(f"FILE_TOO_LARGE: {path}")
    return path

def norm_header(x):
    s = "" if x is None else str(x).strip()
    return re.sub(r"[\s_\-./（）()【】\[\]]+", "", s).lower()

def alias_map(headers):
    norms = {h: norm_header(h) for h in headers if h is not None}
    def find(aliases):
        aa = {norm_header(x) for x in aliases}
        exact = [h for h, n in norms.items() if n in aa]
        if exact:
            return exact[0]
        return None
    return {
        "code": find(CODE_ALIASES),
        "year": find(YEAR_ALIASES),
        "period": find(PERIOD_ALIASES),
        "announce": find(ANNOUNCE_ALIASES),
        "type": find(TYPE_ALIASES),
    }

def parse_code(x):
    if x is None:
        return None
    s = str(x).strip()
    if re.fullmatch(r"\d+(?:\.0+)?", s):
        try:
            return f"{int(float(s)):06d}"[-6:]
        except Exception:
            pass
    m = re.search(r"(?<!\d)(\d{6})(?!\d)", s)
    if m:
        return m.group(1)
    return None

def excel_serial_to_date(v):
    try:
        f = float(v)
    except Exception:
        return None
    if 20000 <= f <= 80000:
        return date(1899, 12, 30) + timedelta(days=int(f))
    return None

def parse_date(x):
    if x is None:
        return None
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    d = excel_serial_to_date(x)
    if d:
        return d
    s = str(x).strip()
    if not s:
        return None
    s = s.replace("年", "-").replace("月", "-").replace("日", "")
    s = s.replace("/", "-").replace(".", "-")
    m = re.search(r"((?:19|20)\d{2})-?(\d{1,2})-?(\d{1,2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    if re.fullmatch(r"(?:19|20)\d{6}", re.sub(r"\D", "", s)):
        z = re.sub(r"\D", "", s)
        try:
            return date(int(z[:4]), int(z[4:6]), int(z[6:8]))
        except ValueError:
            return None
    return None

def parse_year(x):
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        y = int(x)
        return y if 1990 <= y <= 2030 else None
    s = str(x).strip()
    m = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", s)
    return int(m.group(1)) if m else None

def plausible_anchor(y, d):
    if not y or not d:
        return False
    return date(y + 1, 1, 1) <= d <= date(y + 1, 12, 31)

def strongly_plausible_anchor(y, d):
    if not y or not d:
        return False
    return date(y + 1, 1, 1) <= d <= date(y + 1, 6, 30)

def annual_type_ok(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    if any(k in s for k in ("annual", "年度", "年报", "a")):
        # Avoid treating arbitrary strings containing letter a as annual.
        if s == "a" or "annual" in s or "年度" in s or "年报" in s:
            return True
    if any(k in s for k in ("quarter", "季度", "semi", "半年度", "半年")):
        return False
    return None

def annual_period_ok(v):
    d = parse_date(v)
    if d:
        return d.month == 12 and d.day in (30, 31)
    s = "" if v is None else str(v).strip()
    return bool(re.search(r"(?:12[-/.]?31|12月31)", s))

def split_name(year):
    return "train_2010_2018" if year <= 2018 else "validation_2019_2020" if year <= 2020 else "test_2021_2024"

def latest_stage_a():
    root = BASE / "label_v1_stageA"
    if not root.is_dir():
        raise SystemExit("未找到 Stage A 目录。")
    candidates = []
    for d in root.glob("stageA_*"):
        m = d / "manifest_v1_stageA.json"
        if m.is_file():
            try:
                obj = json.loads(m.read_text(encoding="utf-8"))
                if obj.get("completed"):
                    candidates.append((m.stat().st_mtime, d))
            except Exception:
                pass
    if not candidates:
        raise SystemExit("未找到 completed Stage A 输出。")
    return max(candidates)[1]

def verify_stage_a(d: Path):
    d = d.resolve()
    allowed = (BASE / "label_v1_stageA").resolve()
    if d.parent != allowed:
        raise ValueError("STAGE_A_DIR_NOT_DIRECT_CHILD")
    mani_path = checked_file(d / "manifest_v1_stageA.json", 20 * 1024 * 1024)
    mani = json.loads(mani_path.read_text(encoding="utf-8"))
    if not mani.get("completed") or mani.get("final_labels_authorized"):
        raise ValueError("STAGE_A_MANIFEST_INVALID")
    for name, meta in mani.get("outputs", {}).items():
        p = d / name
        checked_file(p, 1024 * 1024 * 1024)
        if sha256(p) != meta.get("sha256"):
            raise ValueError(f"STAGE_A_OUTPUT_HASH_MISMATCH: {name}")
    return mani

def load_panel_keys():
    import pyarrow.parquet as pq
    if sha256(PANEL) != PANEL_SHA256:
        raise ValueError("PANEL_PIN_MISMATCH")
    pf = pq.ParquetFile(PANEL)
    cols = pf.read(columns=["firm_id", "year"], use_threads=False).to_pydict()
    keys = []
    for f, y in zip(cols["firm_id"], cols["year"]):
        keys.append((str(f), int(y)))
    if len(keys) != 51675 or len(set(keys)) != 51675:
        raise ValueError("PANEL_KEY_COUNT_UNEXPECTED")
    return keys

def fresh_output():
    root = BASE / "label_v1_stageB"
    if root.is_symlink():
        raise ValueError("OUTPUT_PARENT_SYMLINK_NOT_ALLOWED")
    root.mkdir(mode=0o700, exist_ok=True)
    d = root / ("stageB_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    d.mkdir(mode=0o700, exist_ok=False)
    return d

def read_parquet_rows(path: Path, columns=None):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=columns, use_threads=False)
    return t.to_pylist()

def write_parquet(path: Path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.Table.from_pylist(list(rows))
    pq.write_table(t, path, compression="zstd", use_dictionary=True)

def csv_headers(path: Path):
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with path.open("r", encoding=enc, errors="strict", newline="") as f:
                sample = f.read(16384)
            if not sample:
                return [], enc, ","
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
                delim = dialect.delimiter
            except Exception:
                delim = "\t" if sample.count("\t") > sample.count(",") else ","
            row = next(csv.reader(io.StringIO(sample), delimiter=delim))
            return row, enc, delim
        except Exception:
            continue
    return [], None, None

def xlsx_shared_strings(z):
    out = []
    try:
        with z.open("xl/sharedStrings.xml") as f:
            root = ET.parse(f).getroot()
        ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        for si in root.findall(ns + "si"):
            texts = [t.text or "" for t in si.iter(ns + "t")]
            out.append("".join(texts))
    except KeyError:
        pass
    return out

def xlsx_sheet_targets(z):
    ns_main = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    ns_rel = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    pkg_rel = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    with z.open("xl/workbook.xml") as f:
        wb = ET.parse(f).getroot()
    rels = {}
    with z.open("xl/_rels/workbook.xml.rels") as f:
        rr = ET.parse(f).getroot()
        for rel in rr.findall(pkg_rel + "Relationship"):
            rels[rel.attrib["Id"]] = rel.attrib["Target"]
    ans = []
    for sh in wb.find(ns_main + "sheets"):
        rid = sh.attrib.get(ns_rel + "id")
        target = rels.get(rid)
        if target:
            if not target.startswith("xl/"):
                target = "xl/" + target.lstrip("/")
            ans.append((sh.attrib.get("name", ""), target))
    return ans

def cell_col(cell_ref):
    m = re.match(r"([A-Z]+)", cell_ref or "")
    if not m:
        return None
    x = 0
    for ch in m.group(1):
        x = x * 26 + (ord(ch) - 64)
    return x - 1

def xlsx_iter_rows(path: Path, max_sheets=3):
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as z:
        shared = xlsx_shared_strings(z)
        sheets = xlsx_sheet_targets(z)[:max_sheets]
        for sheet_name, target in sheets:
            with z.open(target) as f:
                for ev, elem in ET.iterparse(f, events=("end",)):
                    if elem.tag != ns + "row":
                        continue
                    vals = {}
                    for c in elem.findall(ns + "c"):
                        idx = cell_col(c.attrib.get("r"))
                        if idx is None:
                            continue
                        typ = c.attrib.get("t")
                        v = c.find(ns + "v")
                        val = None if v is None else v.text
                        if typ == "s" and val is not None:
                            try:
                                val = shared[int(val)]
                            except Exception:
                                pass
                        elif typ == "inlineStr":
                            isel = c.find(ns + "is")
                            if isel is not None:
                                val = "".join((t.text or "") for t in isel.iter(ns + "t"))
                        vals[idx] = val
                    if vals:
                        mx = max(vals)
                        yield sheet_name, [vals.get(i) for i in range(mx + 1)]
                    elem.clear()

def inspect_headers(path: Path):
    ext = path.suffix.lower()
    try:
        if ext == ".parquet":
            import pyarrow.parquet as pq
            return [("parquet", pq.ParquetFile(path).schema_arrow.names)]
        if ext in (".csv", ".tsv"):
            h, enc, delim = csv_headers(path)
            return [(f"csv:{enc}:{repr(delim)}", h)] if h else []
        if ext == ".xlsx":
            out = []
            gen = xlsx_iter_rows(path)
            seen = set()
            for sheet, row in gen:
                if sheet not in seen:
                    out.append((f"xlsx:{sheet}", row))
                    seen.add(sheet)
                    if len(seen) >= 3:
                        break
            return out
        if ext in (".jsonl", ".ndjson"):
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            return [("jsonl", list(obj))]
                        break
        if ext == ".json" and path.stat().st_size <= 50 * 1024 * 1024:
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return [("json", list(obj[0]))]
            if isinstance(obj, dict):
                # common manifest: list nested under data/items/results
                for k in ("data", "items", "results", "records"):
                    v = obj.get(k)
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        return [(f"json:{k}", list(v[0]))]
    except Exception:
        return []
    return []

def paths_from_stage_a_inventory(stage_a):
    p = stage_a / "anchor_source_inventory_LOCAL_ONLY.txt"
    out = []
    if p.is_file():
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("FILENAME_LEAD | "):
                q = Path(line.split(" | ", 1)[1].strip())
                if q.is_file():
                    out.append(q)
    return out

def discover_candidate_files(stage_a):
    candidates = []
    seen = set()

    def add(p):
        try:
            rp = p.resolve()
            if rp in seen or not rp.is_file():
                return
            if rp.suffix.lower() not in {".parquet", ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xlsx"}:
                return
            if rp.stat().st_size > 300 * 1024 * 1024:
                return
            seen.add(rp)
            candidates.append(rp)
        except OSError:
            pass

    for p in paths_from_stage_a_inventory(stage_a):
        add(p)

    roots = [
        PROJECT / "data/raw/annual_reports",
        PROJECT / "data/interim",
        PROJECT / "data/processed",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        n = 0
        for base, dirs, files in os.walk(root):
            tick()
            rel_depth = len(Path(base).relative_to(root).parts)
            dirs[:] = [d for d in dirs if d not in {".venv", "venv", "__pycache__", "models", "checkpoints", "pdfs"}]
            if rel_depth > 4:
                dirs[:] = []
                continue
            for fn in files:
                n += 1
                if n > 10000:
                    dirs[:] = []
                    break
                if LIKELY_NAME.search(fn) or LIKELY_NAME.search(str(Path(base).name)):
                    add(Path(base) / fn)

    # Bounded CSMAR descriptor-driven discovery. Read descriptors only, then add
    # same-directory tabular files if the descriptor mentions annual/report + announcement/disclosure date.
    if CSMAR_ROOT.is_dir():
        inspected = 0
        for p in CSMAR_ROOT.rglob("*.txt"):
            tick()
            if inspected >= 2500:
                break
            inspected += 1
            try:
                if p.stat().st_size > 5 * 1024 * 1024:
                    continue
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                try:
                    txt = p.read_text(encoding="gb18030", errors="ignore")
                except Exception:
                    continue
            if re.search(r"公告日期|披露日期|发布日期|Annodt|announce|publish|filing", txt, re.I) and \
               re.search(r"年度报告|年报|报告期|Accper|Typrep|会计期间", txt, re.I):
                for q in p.parent.iterdir():
                    if q.is_file() and q.suffix.lower() in {".xlsx", ".csv", ".tsv", ".parquet", ".json", ".jsonl"}:
                        add(q)

    return candidates

def candidate_specs(files):
    specs = []
    for p in files:
        tick()
        for fmt, headers in inspect_headers(p):
            if not headers:
                continue
            amap = alias_map(headers)
            if amap["code"] and amap["announce"] and (amap["year"] or amap["period"]):
                semantic = 0
                if LIKELY_NAME.search(p.name): semantic += 2
                if re.search(r"annual|年报|年度报告", str(p.parent) + "/" + p.name, re.I): semantic += 3
                if amap["period"]: semantic += 2
                if amap["type"]: semantic += 1
                specs.append({
                    "path": str(p),
                    "format": fmt,
                    "headers": headers,
                    "map": amap,
                    "semantic_score": semantic,
                })
    return specs

def _iter_records_for_spec(spec):
    path = Path(spec["path"])
    fmt = spec["format"]
    headers = spec["headers"]
    amap = spec["map"]
    wanted = {v for v in amap.values() if v}
    ext = path.suffix.lower()

    if ext == ".parquet":
        import pyarrow.parquet as pq
        t = pq.read_table(path, columns=[h for h in headers if h in wanted], use_threads=False)
        for r in t.to_pylist():
            yield r
        return

    if ext in (".csv", ".tsv"):
        h, enc, delim = csv_headers(path)
        with path.open("r", encoding=enc, errors="ignore", newline="") as f:
            rd = csv.DictReader(f, delimiter=delim)
            for r in rd:
                yield {k: r.get(k) for k in wanted}
        return

    if ext == ".xlsx":
        target_sheet = fmt.split(":", 1)[1] if ":" in fmt else None
        it = xlsx_iter_rows(path)
        active_headers = None
        index = None
        for sheet, row in it:
            if target_sheet and sheet != target_sheet:
                continue
            if active_headers is None:
                active_headers = ["" if x is None else str(x).strip() for x in row]
                index = {h: i for i, h in enumerate(active_headers)}
                continue
            if not index:
                continue
            yield {k: (row[index[k]] if k in index and index[k] < len(row) else None) for k in wanted}
        return

    if ext in (".jsonl", ".ndjson"):
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if isinstance(r, dict):
                        yield {k: r.get(k) for k in wanted}
        return

    if ext == ".json":
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(obj, dict):
            for k in ("data", "items", "results", "records"):
                if isinstance(obj.get(k), list):
                    obj = obj[k]
                    break
        if isinstance(obj, list):
            for r in obj:
                if isinstance(r, dict):
                    yield {k: r.get(k) for k in wanted}
        return

def build_anchor_map(spec, panel_set):
    amap = spec["map"]
    mapping = {}
    duplicate_dates = 0
    rows_seen = 0
    usable = 0
    type_filtered = 0
    period_filtered = 0

    for r in _iter_records_for_spec(spec):
        tick()
        rows_seen += 1
        if rows_seen > 2_000_000:
            break

        code = parse_code(r.get(amap["code"]))
        if not code:
            continue
        d = parse_date(r.get(amap["announce"]))
        if not d or d > CUTOFF:
            continue

        y = parse_year(r.get(amap["year"])) if amap["year"] else None
        period_val = r.get(amap["period"]) if amap["period"] else None
        if y is None and period_val is not None:
            y = parse_year(period_val)
        if y is None or not 2010 <= y <= 2024:
            continue

        if amap["type"]:
            ok = annual_type_ok(r.get(amap["type"]))
            if ok is False:
                type_filtered += 1
                continue

        if amap["period"]:
            pd = parse_date(period_val)
            if pd is not None and not (pd.month == 12 and pd.day in (30, 31)):
                period_filtered += 1
                continue

        key = (code, y)
        if key not in panel_set:
            continue
        usable += 1
        old = mapping.get(key)
        if old is None or d < old:
            if old is not None and d != old:
                duplicate_dates += 1
            mapping[key] = d
        elif d != old:
            duplicate_dates += 1

    return mapping, {
        "rows_seen": rows_seen,
        "usable_panel_rows": usable,
        "duplicate_distinct_dates": duplicate_dates,
        "type_filtered": type_filtered,
        "period_filtered": period_filtered,
    }

def score_anchor_map(spec, mapping, panel_keys, stats):
    n = len(panel_keys)
    cov = len(mapping) / n
    vals = [(y, d) for (f, y), d in mapping.items()]
    plausible = sum(plausible_anchor(y, d) for y, d in vals) / len(vals) if vals else 0.0
    strong = sum(strongly_plausible_anchor(y, d) for y, d in vals) / len(vals) if vals else 0.0
    preyear = sum(d <= date(y,12,31) for y,d in vals)
    score = cov * 100 + plausible * 15 + strong * 5 + spec["semantic_score"]
    return {
        "path": spec["path"],
        "format": spec["format"],
        "code_col": spec["map"]["code"],
        "year_col": spec["map"]["year"],
        "period_col": spec["map"]["period"],
        "announce_col": spec["map"]["announce"],
        "type_col": spec["map"]["type"],
        "semantic_score": spec["semantic_score"],
        "panel_anchor_count": len(mapping),
        "coverage": cov,
        "plausible_t_plus_1": plausible,
        "strong_jan_to_jun_t_plus_1": strong,
        "anchor_on_or_before_fiscal_year_end": preyear,
        "candidate_score": score,
        **stats,
    }

def verify_descriptor_date_semantics():
    if sha256(DESCRIPTOR) != DESCRIPTOR_SHA256:
        raise ValueError("DESCRIPTOR_PIN_MISMATCH")
    text = DESCRIPTOR.read_text(encoding="utf-8", errors="ignore")
    if not text:
        try:
            text = DESCRIPTOR.read_text(encoding="gb18030", errors="ignore")
        except Exception:
            text = ""
    lines = text.splitlines()
    snippets = {}
    for field in ("DeclareDate", "DisposalDate"):
        hits = []
        for i, line in enumerate(lines):
            if field.lower() in line.lower():
                chunk = " ".join(lines[max(0,i-1):min(len(lines),i+2)])
                hits.append(chunk[:500])
        snippets[field] = hits[:5]
    joined_decl = " ".join(snippets["DeclareDate"])
    joined_disp = " ".join(snippets["DisposalDate"])
    declare_public_semantics = bool(re.search(r"公告|披露|公布|发布|declare|announcement|disclosure", joined_decl, re.I))
    disposal_decision_semantics = bool(re.search(r"处罚|处分|处理|决定|disposal|punish|decision", joined_disp, re.I))
    return {
        "declare_hits": len(snippets["DeclareDate"]),
        "disposal_hits": len(snippets["DisposalDate"]),
        "declare_public_semantics_supported": declare_public_semantics,
        "disposal_decision_semantics_supported": disposal_decision_semantics,
        # snippets stay local in protocol; report only booleans/counts
        "_snippets": snippets,
    }

def quantiles(vals):
    vals = sorted(vals)
    if not vals:
        return {}
    def q(p):
        x = (len(vals)-1)*p
        lo = math.floor(x); hi = math.ceil(x)
        if lo == hi: return vals[lo]
        return vals[lo]*(hi-x)+vals[hi]*(x-lo)
    return {
        "n": len(vals),
        "min": vals[0],
        "p25": round(q(.25),1),
        "median": round(q(.5),1),
        "p75": round(q(.75),1),
        "max": vals[-1],
    }

def reveal_in_finder(path: Path):
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", "-R", str(path)], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def main():
    global START
    START = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-b", action="store_true")
    ap.add_argument("--stage-a-dir", type=Path, default=None)
    ap.add_argument("--reveal", action="store_true")
    args = ap.parse_args()
    if not args.stage_b:
        ap.error("--stage-b is required")

    os.umask(0o077)
    os.chdir(Path.home())
    out = fresh_output()
    report_path = out / "label_v1_stageB_report.txt"

    try:
        print("[1/7] 核验 Stage A、固定面板与来源哈希……", flush=True)
        stage_a = (args.stage_a_dir or latest_stage_a()).resolve()
        stage_a_manifest = verify_stage_a(stage_a)
        panel_keys = load_panel_keys()
        panel_set = set(panel_keys)

        print("[2/7] 只读发现本地年报披露日期候选源并检查字段结构……", flush=True)
        candidate_files = discover_candidate_files(stage_a)
        specs = candidate_specs(candidate_files)

        print(f"      找到 {len(specs)} 个具有 code + year/period + announcement/filing date 字段的候选表。", flush=True)

        print("[3/7] 逐个候选源计算面板覆盖率与 t+1 日期合理性……", flush=True)
        scored = []
        anchor_maps = {}
        for i, spec in enumerate(specs, 1):
            try:
                mapping, st = build_anchor_map(spec, panel_set)
                sc = score_anchor_map(spec, mapping, panel_keys, st)
                scored.append(sc)
                anchor_maps[(spec["path"], spec["format"])] = mapping
                print(f"      {i}/{len(specs)} coverage={sc['coverage']:.3%} "
                      f"plausible={sc['plausible_t_plus_1']:.3%} :: {Path(spec['path']).name}", flush=True)
            except Exception as e:
                scored.append({
                    "path": spec["path"], "format": spec["format"],
                    "error": f"{type(e).__name__}:{e}",
                    "coverage": 0.0, "plausible_t_plus_1": 0.0,
                    "candidate_score": -1,
                })

        scored.sort(key=lambda x: x.get("candidate_score", -1), reverse=True)

        with (out / "anchor_candidates_v1_stageB_LOCAL_ONLY.csv").open("w", encoding="utf-8", newline="") as f:
            fields = sorted({k for r in scored for k in r.keys()})
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in scored:
                w.writerow(r)

        selected = None
        if scored:
            top = scored[0]
            if (top.get("coverage", 0) >= 0.90 and
                top.get("plausible_t_plus_1", 0) >= 0.95 and
                top.get("anchor_on_or_before_fiscal_year_end", 1) == 0):
                selected = top

        if selected is None:
            lines = [
                "PeerJ 141707 | Label v1 Stage B",
                "Status intent: post-filing anchor integration; NO FINAL LABELS; NO TRAINING",
                "Local time: " + datetime.now().astimezone().isoformat(),
                "Stage A: " + str(stage_a),
                "Output: " + str(out),
                "",
                "A. SAFETY",
                "stage_a_verified=true",
                "panel_sha256_verified=true",
                "source_files_modified=false",
                "training_or_prediction_run=false",
                "final_nonnull_labels_written=0",
                "",
                "B. ANNUAL-REPORT ANCHOR DISCOVERY",
                f"candidate_files_examined={len(candidate_files)}",
                f"schema_compatible_candidates={len(specs)}",
                "automatic_anchor_source_selected=false",
            ]
            if scored:
                t = scored[0]
                lines += [
                    f"best_candidate_path={t.get('path')}",
                    f"best_candidate_format={t.get('format')}",
                    f"best_candidate_coverage={t.get('coverage',0):.6f}",
                    f"best_candidate_plausible_t_plus_1={t.get('plausible_t_plus_1',0):.6f}",
                    f"best_candidate_strong_jan_to_jun={t.get('strong_jan_to_jun_t_plus_1',0):.6f}",
                ]
            lines += [
                "",
                "C. STOP CONDITION",
                "No local anchor source met the conservative auto-selection gate (coverage>=90%, t+1 plausibility>=95%, no pre-fiscal-year-end anchors).",
                "No post-filing status or final label was inferred.",
                "",
                "STATUS: LABEL_V1_STAGE_B_STOP_ANCHOR_SOURCE_REQUIRED",
                "Upload only label_v1_stageB_report.txt and keep LOCAL_ONLY files local.",
            ]
            report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print("STATUS: LABEL_V1_STAGE_B_STOP_ANCHOR_SOURCE_REQUIRED")
            print("请上传：", report_path)
            if args.reveal:
                reveal_in_finder(report_path)
            return

        print("[4/7] 冻结选中的年报首次披露锚点（仍不生成最终标签）……", flush=True)
        sel_key = (selected["path"], selected["format"])
        anchors = anchor_maps[sel_key]
        anchor_rows = []
        bypart = Counter()
        follow_by_year = defaultdict(list)
        for firm, year in panel_keys:
            d = anchors.get((firm, year))
            if d:
                bypart[split_name(year)] += 1
                fd = (CUTOFF - d).days
                follow_by_year[year].append(fd)
            anchor_rows.append({
                "company_code": firm,
                "fiscal_year": year,
                "partition": split_name(year),
                "annual_report_filing_anchor": d.isoformat() if d else None,
                "anchor_source_path": selected["path"],
                "anchor_source_format": selected["format"],
                "anchor_status": "verified_local_metadata_source" if d else "missing_anchor",
            })
        write_parquet(out / "annual_report_anchors_v1_stageB.parquet", anchor_rows)

        print("[5/7] 核验 CSMAR 日期字段语义，并建立 earliest qualifying public-sanction date……", flush=True)
        sem = verify_descriptor_date_semantics()
        if not sem["declare_public_semantics_supported"]:
            # Do not infer public dates if descriptor semantics are not sufficiently explicit.
            sanction_public_authorized = False
        else:
            sanction_public_authorized = True

        events = read_parquet_rows(stage_a / "event_ledger_v1_stageA.parquet")
        event_by_ref = {r["row_ref"]: r for r in events}
        pairs = read_parquet_rows(stage_a / "pair_ledger_v1_stageA.parquet")
        fy = read_parquet_rows(stage_a / "firm_year_support_v1_stageA.parquet")

        anchor_dict = {(r["company_code"], r["fiscal_year"]): r["annual_report_filing_anchor"] for r in anchor_rows}
        qualifying_by_scope_key = {"strict": defaultdict(list), "loose": defaultdict(list)}
        pair_out = []
        timing_counts = {"strict": Counter(), "loose": Counter()}

        for p in pairs:
            scope = p["scope"]
            status = p["preanchor_status"]
            firm = p.get("company_code")
            year = p.get("violation_year")
            if isinstance(year, str) and year.isdigit():
                year = int(year)
            key = (firm, year) if firm and isinstance(year, int) else None
            anchor_s = anchor_dict.get(key) if key else None
            anchor_d = parse_date(anchor_s)

            ev = event_by_ref.get(p["row_ref"], {})
            public_d = parse_date(ev.get("declare_date")) if sanction_public_authorized else None
            decision_d = parse_date(ev.get("disposal_date"))

            if status not in {"A_text_year_supported_pre_anchor", "B_raw_year_only_pre_anchor"}:
                tstat = "not_preanchor_supported_candidate"
            elif not anchor_d:
                tstat = "unresolved_missing_annual_report_anchor"
            elif not public_d:
                tstat = "unresolved_missing_or_unverified_public_sanction_date"
            elif public_d > anchor_d:
                tstat = "post_filing_supported"
                qualifying_by_scope_key[scope][key].append(public_d)
            else:
                tstat = "sanction_public_on_or_before_annual_filing"
            timing_counts[scope][tstat] += 1

            q = dict(p)
            q["annual_report_filing_anchor"] = anchor_d.isoformat() if anchor_d else None
            q["qualifying_public_sanction_date"] = public_d.isoformat() if public_d else None
            q["decision_or_disposal_date"] = decision_d.isoformat() if decision_d else None
            q["post_filing_status"] = tstat
            q["final_label"] = None
            q["training_ready"] = False
            pair_out.append(q)

        write_parquet(out / "pair_postfiling_v1_stageB.parquet", pair_out)

        print("[6/7] 聚合 firm-year timing 状态、legacy 对照与成熟度，不赋0/1……", flush=True)
        fy_out = []
        legacy_counts = {"strict": Counter(), "loose": Counter()}
        review_rows = []
        for r in fy:
            scope = r["scope"]
            key = (r["company_code"], int(r["fiscal_year"]))
            anchor_d = parse_date(anchor_dict.get(key))
            dates = qualifying_by_scope_key[scope].get(key, [])
            earliest = min(dates) if dates else None
            if earliest and anchor_d:
                state = "post_filing_supported"
            elif r["preanchor_evidence_state"] in {"A_text_year_supported_pre_anchor", "B_raw_year_only_pre_anchor"}:
                # Inspect pair-level statuses for this key/scope.
                sts = [x["post_filing_status"] for x in pair_out
                       if x.get("scope") == scope and x.get("company_code") == key[0]
                       and int(x.get("violation_year") or -1) == key[1]]
                if "sanction_public_on_or_before_annual_filing" in sts:
                    state = "pre_or_same_day_sanction_no_postfiling_support"
                elif not anchor_d:
                    state = "unresolved_missing_annual_report_anchor"
                else:
                    state = "unresolved_missing_or_unverified_public_sanction_date"
            elif r["preanchor_evidence_state"] == "unresolved_pre_anchor":
                state = "unresolved_preanchor_evidence"
            else:
                state = "no_machine_support_not_negative"

            follow = (CUTOFF - anchor_d).days if anchor_d else None
            legacy = int(r["legacy_label"])
            legacy_counts[scope][f"legacy{legacy}__{state}"] += 1

            o = dict(r)
            o["annual_report_filing_anchor"] = anchor_d.isoformat() if anchor_d else None
            o["earliest_qualifying_public_sanction_date"] = earliest.isoformat() if earliest else None
            o["post_filing_evidence_state"] = state
            o["followup_days_to_cutoff"] = follow
            o["final_label"] = None
            o["training_ready"] = False
            fy_out.append(o)

            if (legacy == 1 and state != "post_filing_supported") or \
               (legacy == 0 and state == "post_filing_supported") or \
               state.startswith("unresolved"):
                review_rows.append({
                    "company_code": key[0],
                    "fiscal_year": key[1],
                    "partition": r["partition"],
                    "scope": scope,
                    "legacy_label": legacy,
                    "preanchor_evidence_state": r["preanchor_evidence_state"],
                    "post_filing_evidence_state": state,
                    "annual_report_filing_anchor": o["annual_report_filing_anchor"],
                    "earliest_qualifying_public_sanction_date": o["earliest_qualifying_public_sanction_date"],
                    "followup_days_to_cutoff": follow,
                })

        write_parquet(out / "firm_year_postfiling_v1_stageB.parquet", fy_out)

        with (out / "review_queue_stageB_LOCAL_ONLY.csv").open("w", encoding="utf-8", newline="") as f:
            fields = list(review_rows[0]) if review_rows else ["company_code","fiscal_year","scope"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in review_rows:
                x = dict(r); x["company_code"] = "'" + x["company_code"]
                w.writerow(x)

        print("[7/7] 写出可上传的聚合报告与不可变 manifest……", flush=True)
        maturity = {str(y): quantiles(v) for y, v in sorted(follow_by_year.items())}
        protocol = {
            "version": VERSION,
            "stage_a_dir": str(stage_a),
            "annual_anchor_selection_gate": {
                "coverage_min": 0.90,
                "t_plus_1_plausibility_min": 0.95,
                "pre_fiscal_year_end_anchor_max": 0,
            },
            "selected_anchor_source": selected,
            "csrc_date_field_semantics": sem,
            "public_sanction_date_rule": "DeclareDate only when descriptor supports public/announcement/disclosure semantics",
            "decision_date_rule": "DisposalDate retained separately; never substituted for public date",
            "post_filing_rule": "qualifying public-sanction date must be strictly later than annual-report filing anchor",
            "negative_rule": "absence of qualifying evidence is not a negative label in Stage B",
            "maturity_rule": "follow-up days measured from annual-report filing anchor to fixed cutoff 2026-05-08; no maturity threshold frozen in Stage B",
            "final_labels_authorized": False,
            "training_authorized": False,
        }
        (out / "protocol_v1_stageB.json").write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8"
        )

        report = []
        report += [
            "PeerJ 141707 | Label v1 Stage B annual-report anchors + post-filing timing",
            "Status intent: timing evidence integration; NO FINAL LABELS; NO TRAINING",
            "Local time: " + datetime.now().astimezone().isoformat(),
            "Stage A: " + str(stage_a),
            "Output: " + str(out),
            "",
            "A. INPUT / SAFETY",
            "stage_a_manifest_verified=true",
            f"panel_sha256={PANEL_SHA256}",
            f"descriptor_sha256={DESCRIPTOR_SHA256}",
            "source_files_modified=false",
            "training_or_prediction_run=false",
            "final_nonnull_labels_written=0",
            "",
            "B. SELECTED ANNUAL-REPORT FILING ANCHOR SOURCE",
            "path=" + selected["path"],
            "format=" + selected["format"],
            "code_col=" + str(selected.get("code_col")),
            "year_col=" + str(selected.get("year_col")),
            "period_col=" + str(selected.get("period_col")),
            "announce_col=" + str(selected.get("announce_col")),
            "type_col=" + str(selected.get("type_col")),
            f"panel_anchor_count={selected['panel_anchor_count']}",
            f"coverage={selected['coverage']:.6f}",
            f"plausible_t_plus_1={selected['plausible_t_plus_1']:.6f}",
            f"strong_jan_to_jun_t_plus_1={selected['strong_jan_to_jun_t_plus_1']:.6f}",
            f"duplicate_distinct_dates={selected.get('duplicate_distinct_dates',0)}",
            "partition_anchor_counts=" + json.dumps(dict(bypart), ensure_ascii=False, sort_keys=True),
            "",
            "C. CSRC DATE FIELD SEMANTICS",
            f"DeclareDate_descriptor_hits={sem['declare_hits']}",
            f"DisposalDate_descriptor_hits={sem['disposal_hits']}",
            f"DeclareDate_public_semantics_supported={str(sem['declare_public_semantics_supported']).lower()}",
            f"DisposalDate_decision_semantics_supported={str(sem['disposal_decision_semantics_supported']).lower()}",
            "",
            "D. TYPE-YEAR PAIR TIMING STATUS",
            "strict=" + json.dumps(dict(timing_counts["strict"]), ensure_ascii=False, sort_keys=True),
            "loose=" + json.dumps(dict(timing_counts["loose"]), ensure_ascii=False, sort_keys=True),
            "",
            "E. LEGACY vs STAGE-B TIMING EVIDENCE (NOT ERROR ADJUDICATION)",
            "strict=" + json.dumps(dict(legacy_counts["strict"]), ensure_ascii=False, sort_keys=True),
            "loose=" + json.dumps(dict(legacy_counts["loose"]), ensure_ascii=False, sort_keys=True),
            "",
            "F. FOLLOW-UP MATURITY TO FIXED CUTOFF 2026-05-08",
            json.dumps(maturity, ensure_ascii=False, sort_keys=True),
            "",
            "G. REVIEW WORKLOAD",
            f"review_queue_firm_year_scope_rows={len(review_rows)}",
            "Queue is priority adjudication, not an estimated error rate.",
            "",
            "H. NEXT GATE",
            "Stage B does not freeze a final 0/1 label or a mature-test window.",
            "Next step is evidence adjudication + maturity-rule freeze using these timing results.",
            "",
            "STATUS: LABEL_V1_STAGE_B_COMPLETE_READY_FOR_ADJUDICATION",
            "Upload only label_v1_stageB_report.txt. Keep LOCAL_ONLY and parquet files local.",
        ]
        report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

        files = {}
        for p in sorted(out.iterdir()):
            if p.is_file():
                files[p.name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
        mani = {
            "version": VERSION,
            "completed": True,
            "created_at": datetime.now().astimezone().isoformat(),
            "stage_a_manifest_sha256": sha256(stage_a / "manifest_v1_stageA.json"),
            "outputs": files,
            "final_labels_authorized": False,
            "training_authorized": False,
        }
        (out / "manifest_v1_stageB.json").write_text(
            json.dumps(mani, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8"
        )

        print("STATUS: LABEL_V1_STAGE_B_COMPLETE_READY_FOR_ADJUDICATION")
        print("请只上传：")
        print(report_path)
        if args.reveal:
            reveal_in_finder(report_path)

    except Exception as e:
        # Create a small failure report in the new output directory only.
        lines = [
            "PeerJ 141707 | Label v1 Stage B FAILED SAFELY",
            "Local time: " + datetime.now().astimezone().isoformat(),
            "Output: " + str(out),
            f"ERROR_TYPE={type(e).__name__}",
            f"ERROR={e}",
            "source_files_modified=false",
            "training_or_prediction_run=false",
            "final_nonnull_labels_written=0",
            "STATUS: LABEL_V1_STAGE_B_FAILED_SAFE",
        ]
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), file=sys.stderr)
        if args.reveal:
            reveal_in_finder(report_path)
        raise

if __name__ == "__main__":
    main()
