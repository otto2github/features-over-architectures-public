#!/usr/bin/env python3
"""PeerJ 141707 — Label v1 Stage A: traceable full-source reconstruction before post-filing anchoring.

Purpose
-------
This is a LOCAL, NO-TRAINING reconstruction step that consumes the already-built
traceable candidate package and the pinned original inputs. It does not overwrite
legacy labels and it does not create final 0/1 targets.

It refines four dimensions that the earlier conservative routing intentionally did
not adjudicate:
  1) regulator class (CSRC HQ / CSRC branch / exchange / other),
  2) document stage (final administrative penalty vs prior notice vs regulatory measure),
  3) listed-company subject evidence,
  4) raw violation-year consistency against explicit report-year wording in Activity.

Final post-filing eligibility is intentionally NOT decided here because a verified
annual-report filing anchor and earliest qualifying public-sanction date are still
required. The output is a draft evidence ledger + legacy-diff triage for Stage B.

No network. No model import. No training. No source mutation. New outputs only under
~/peerj_141707_audit/label_v1_stageA/.
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
import re
import stat
import sys
import time
import uuid

VERSION = "label_v1_stageA_1.0"
BASE = Path.home() / "peerj_141707_audit"
PROJECT = Path(os.environ.get("PEERJ_SOURCE_PROJECT_ROOT", "private_inputs/project"))
RAW_DIR = Path(os.environ.get("PEERJ_VIOLATION_SOURCE_DIR", "private_inputs/csmar/violation"))
DEFAULT_PACKAGE = BASE / "candidates" / "traceable_candidates_20260916_225824_3c596c5a"

PINS = {
    "panel": (PROJECT / "data/processed/kg/node_features_v1_1.parquet",
              "adc00129f06ffb0995691c6d1ba98f740c53561c1e0cc4e91d927b84eb36beb2"),
    "legacy_label_table": (PROJECT / "data/processed/labels/fraud_labels_v0_8.parquet",
                           "af99c86bd452f54d7e79990e9fd9c6b804cf1ac45d6b6b7e81eba98764a724ac"),
    "descriptor": (RAW_DIR / "STK_Violation_Main[DES][xlsx].txt",
                   "14c4cd27901ee9e4ce629c869b529c8d7ae831df8157dc76d30733bebea55e88"),
    "xlsx": (RAW_DIR / "STK_Violation_Main.xlsx",
             "232371d6635f9d65ec7f770c628c44e1f1cb4e158001ae6f6c0ec521916f69aa"),
}

STRICT = frozenset({"P2501", "P2502", "P2503", "P2507"})
LOOSE = STRICT | frozenset({"P2504", "P2505", "P2506", "P2510"})
LABEL_COLS = ("fraud_v07", "fraud_v08_strict", "fraud_v08_loose")
CUTOFF = "2026-05-08"
START = time.monotonic()
MAX_SECONDS = 900


def tick():
    if time.monotonic() - START > MAX_SECONDS:
        raise RuntimeError("TIME_LIMIT_900_SECONDS")


def txt(x):
    return "" if x is None else str(x).strip().replace("\ufeff", "")


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def checked_file(path: Path, max_bytes=512 * 1024 * 1024):
    if not path.is_absolute():
        raise ValueError("ABSOLUTE_PATH_REQUIRED")
    for p in (path, *path.parents):
        if p.is_symlink():
            raise ValueError("SYMLINK_NOT_ALLOWED")
    st = path.stat()
    if not stat.S_ISREG(st.st_mode) or st.st_size > max_bytes:
        raise ValueError("INPUT_NOT_REGULAR_OR_TOO_LARGE")
    return path


def verify_pins():
    out = {}
    for name, (path, expected) in PINS.items():
        checked_file(path)
        got = sha256(path)
        if got != expected:
            raise ValueError(f"PIN_MISMATCH_{name.upper()}")
        out[name] = {"path": str(path), "sha256": got, "bytes": path.stat().st_size}
    return out


def ensure_package(pkg: Path):
    pkg = pkg.resolve()
    expected_parent = (BASE / "candidates").resolve()
    if pkg.parent != expected_parent:
        raise ValueError("PACKAGE_MUST_BE_DIRECT_CHILD_OF_CANDIDATES")
    needed = ["event_evidence.jsonl", "record_candidates.jsonl", "manifest.json", "protocol.json"]
    for n in needed:
        checked_file(pkg / n, max_bytes=300 * 1024 * 1024)
    manifest = json.loads((pkg / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("completed"):
        raise ValueError("CANDIDATE_PACKAGE_NOT_COMPLETE")
    x = manifest.get("input_files", {}).get("xlsx", {}).get("sha256")
    p = manifest.get("input_files", {}).get("panel", {}).get("sha256")
    if x != PINS["xlsx"][1] or p != PINS["panel"][1]:
        raise ValueError("CANDIDATE_PACKAGE_INPUT_HASH_MISMATCH")
    return manifest


def load_panel(path: Path):
    import pyarrow.parquet as pq
    cols = ["firm_id", "year", *LABEL_COLS]
    pf = pq.ParquetFile(path)
    if not set(cols).issubset(pf.schema_arrow.names):
        raise ValueError("PANEL_FIELDS_MISSING")
    d = pf.read(columns=cols, use_threads=False).to_pydict()
    panel = {}
    for i, (firm, year) in enumerate(zip(d["firm_id"], d["year"])):
        if not isinstance(firm, str) or not re.fullmatch(r"\d{6}", firm):
            raise ValueError("PANEL_FIRM_INVALID")
        if type(year) is not int or not 2010 <= year <= 2024:
            raise ValueError("PANEL_YEAR_INVALID")
        key = (firm, year)
        if key in panel:
            raise ValueError("PANEL_DUPLICATE_KEY")
        vals = tuple(int(d[c][i]) for c in LABEL_COLS)
        if any(v not in (0, 1) for v in vals):
            raise ValueError("PANEL_LABEL_INVALID")
        panel[key] = vals
    if len(panel) != 51675:
        raise ValueError("PANEL_ROWCOUNT_UNEXPECTED")
    return panel


def norm_compact(s):
    return re.sub(r"[\s　\u00a0*＊ＳＴstST（）()\[\]【】]", "", txt(s))


def authority_class(supervisor, promulgator):
    s = txt(supervisor)
    p = txt(promulgator)
    z = s + "|" + p
    if re.search(r"证券交易所|上交所|深交所|中小板公司管理部|创业板公司管理部", z):
        return "exchange"
    if s in {"中国证监会", "中国证券监督管理委员会"}:
        return "csrc_hq"
    if re.search(r"(?:中国证监会|中国证券监督管理委员会).*(?:监管局|证监局)", z) or re.search(r"(?:^|[（(])[^）)]*证监局[）)]?$", s) or re.search(r"证监局", s):
        return "csrc_branch"
    if re.search(r"中国证监会|中国证券监督管理委员会", z):
        return "csrc_hq_or_affiliated_unresolved"
    return "other_or_mixed"


def document_stage(title, measure, auth):
    t = txt(title)
    m = txt(measure)
    z = t + "\n" + m
    if re.search(r"事先告知|告知书|拟决定|拟对.*处罚", z):
        return "prior_notice"
    if auth == "exchange" or re.search(r"监管函|纪律处分|公开谴责|通报批评", t):
        return "exchange_measure"
    if re.search(r"行政监管措施|监管措施决定|警示函|责令整改|责令改正措施", t) or (
        re.search(r"采取.*(?:警示函|责令改正).*监管措施", m) and not re.search(r"行政处罚决定", t)
    ):
        return "regulatory_measure"
    final_title = bool(re.search(r"行政处罚决定(?:书)?", t))
    final_text = bool(re.search(r"(?:我会|我局|本会|本局)决定[:：]", m)) and bool(re.search(r"罚款|给予警告|没收|责令改正", m))
    market = bool(re.search(r"市场禁入决定(?:书)?", t))
    if final_title or final_text:
        return "final_penalty_and_market_ban" if market else "final_penalty"
    if market:
        return "market_ban_only_or_mixed"
    if re.search(r"立案|调查通知", t):
        return "investigation"
    return "unresolved_document_stage"


def parse_p26_codes(value):
    s = txt(value).upper()
    return sorted(set(re.findall(r"P26(?:0[1-7]|99)", s)))


def company_subject_status(raw, stage):
    measure = norm_compact(raw.get("PunishmentMeasure"))
    fullname = norm_compact(raw.get("CoFullName"))
    short = norm_compact(raw.get("ShortName"))
    p26 = parse_p26_codes(raw.get("PunishmentTypeID"))
    if stage.startswith("final_penalty"):
        if fullname and fullname in measure:
            return "confirmed_fullname_in_final_measure"
        if short and len(short) >= 2 and short in measure:
            return "confirmed_shortname_in_final_measure"
        # This field is documented by CSMAR as the listed-company punishment field.
        if p26:
            return "supported_by_company_punishment_field"
        if re.search(r"对(?:你|该)?公司(?:责令改正|给予警告|处以|罚款)", measure):
            return "probable_you_company_final_measure"
        return "unresolved_company_subject"
    return "not_applicable_nonfinal"


def _years(s):
    return [int(x) for x in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", s)]


def _expand_range(a, b):
    a, b = int(a), int(b)
    if 1990 <= a <= b <= 2030 and b - a <= 15:
        return list(range(a, b + 1))
    return []


def extract_report_years(activity):
    """Conservative report-year evidence; does not infer from generic transaction dates."""
    s = txt(activity).replace("—", "-").replace("–", "-").replace("～", "-")
    annual, semi, quarter = set(), set(), set()

    # Explicit ranges tightly tied to annual-report wording.
    for a, b in re.findall(r"((?:19|20)\d{2})\s*年?\s*(?:至|到|-)\s*((?:19|20)\d{2})\s*年(?:度报告|年度报告|年报)", s):
        annual.update(_expand_range(a, b))

    # Lists/ranges immediately preceding annual report wording, e.g. 2018年、2019年年度报告.
    for m in re.finditer(r"((?:(?:19|20)\d{2}\s*年?\s*(?:、|,|，|和|及|至|到|-)?\s*){1,10})(?:年度报告|年报)", s):
        frag = m.group(1)
        ys = _years(frag)
        if re.search(r"至|到|-", frag) and len(ys) >= 2:
            annual.update(_expand_range(ys[0], ys[-1]))
        else:
            annual.update(ys)

    for y in re.findall(r"((?:19|20)\d{2})\s*年(?:半年度报告|半年报)", s):
        semi.add(int(y))
    for y in re.findall(r"((?:19|20)\d{2})\s*年(?:第?[一二三四1234]季度报告|一季报|三季报)", s):
        quarter.add(int(y))

    # Quoted report titles sometimes omit the first 年 before 年度报告 after punctuation normalization.
    for y in re.findall(r"《\s*((?:19|20)\d{2})\s*年(?:年度报告|半年度报告|第?[一二三四1234]季度报告)\s*》", s):
        y = int(y)
        # Already captured by specific regex in most cases; annual is the least harmful fallback.
        if y not in semi and y not in quarter:
            annual.add(y)

    all_periodic = annual | semi | quarter
    return {
        "annual_report_years": sorted(y for y in annual if 1990 <= y <= 2030),
        "semiannual_report_years": sorted(y for y in semi if 1990 <= y <= 2030),
        "quarterly_report_years": sorted(y for y in quarter if 1990 <= y <= 2030),
        "periodic_report_years": sorted(y for y in all_periodic if 1990 <= y <= 2030),
    }


def year_evidence_status(raw_year, report_years):
    if raw_year is None:
        return "raw_year_unresolved"
    ys = set(report_years.get("periodic_report_years", []))
    if not ys:
        return "raw_year_only_no_periodic_report_year_extracted"
    if raw_year in ys:
        return "raw_year_supported_by_periodic_report_text"
    return "raw_year_conflicts_with_periodic_report_text"


def split_name(year):
    return "train_2010_2018" if year <= 2018 else "validation_2019_2020" if year <= 2020 else "test_2021_2024"


def preanchor_status(scope, code, year, auth, stage, subject, ystatus):
    scope_codes = STRICT if scope == "strict" else LOOSE
    if code not in scope_codes:
        return "outside_type_scope"
    if year is None or not 2010 <= year <= 2024:
        return "unresolved_year"
    if auth not in {"csrc_hq", "csrc_branch"}:
        return "non_csrc_or_authority_unresolved"
    if not stage.startswith("final_penalty"):
        return "nonfinal_document"
    if subject not in {
        "confirmed_fullname_in_final_measure",
        "confirmed_shortname_in_final_measure",
        "supported_by_company_punishment_field",
        "probable_you_company_final_measure",
    }:
        return "company_subject_unresolved"
    if ystatus == "raw_year_conflicts_with_periodic_report_text":
        return "year_conflict_requires_review"
    if ystatus == "raw_year_supported_by_periodic_report_text":
        return "A_text_year_supported_pre_anchor"
    return "B_raw_year_only_pre_anchor"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            tick()
            if line.strip():
                yield json.loads(line)


def fresh_output():
    parent = BASE / "label_v1_stageA"
    if parent.is_symlink():
        raise ValueError("OUTPUT_PARENT_SYMLINK_NOT_ALLOWED")
    parent.mkdir(mode=0o700, exist_ok=True)
    out = parent / ("stageA_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    out.mkdir(mode=0o700, exist_ok=False)
    return out


def write_parquet(path: Path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = list(rows)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd", use_dictionary=True)
    return len(rows)


def anchor_inventory(project: Path):
    lines = []
    roots = [project / "data/raw/annual_reports", project / "scripts"]
    lines.append("Anchor-source inventory is filename/text-lead only; no PDF body read.")
    for root in roots:
        if not root.exists():
            lines.append(f"MISSING_ROOT | {root}")
            continue
        lines.append(f"ROOT | {root}")
        seen = 0
        hits = 0
        for base, dirs, files in os.walk(root):
            tick()
            rel_depth = len(Path(base).relative_to(root).parts)
            dirs[:] = [d for d in dirs if d not in {".venv", "venv", "__pycache__", "models", "checkpoints"}]
            if rel_depth > 3:
                dirs[:] = []
                continue
            for fn in files:
                seen += 1
                if seen > 6000:
                    lines.append("ENUMERATION_STOP_AFTER_6000_ENTRIES")
                    dirs[:] = []
                    break
                low = fn.lower()
                if any(k in low for k in ("meta", "manifest", "index", "date", "filing", "announce", "report_list")) and low.endswith((".csv", ".json", ".jsonl", ".parquet", ".txt", ".md", ".py")):
                    p = Path(base) / fn
                    lines.append(f"FILENAME_LEAD | {p}")
                    hits += 1
                    if hits >= 120:
                        break
            if hits >= 120 or seen > 6000:
                break
    # Small script grep for likely filing-date symbols.
    scripts = project / "scripts"
    patterns = re.compile(r"ann(?:ounce)?_date|publish_date|filing_date|report_date|公告日期|披露日期|首次披露", re.I)
    if scripts.exists():
        emitted = 0
        for p in scripts.rglob("*.py"):
            tick()
            if any(x in p.parts for x in (".venv", "venv", "__pycache__")):
                continue
            try:
                if p.stat().st_size > 2 * 1024 * 1024:
                    continue
                for n, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if patterns.search(line):
                        lines.append(f"SCRIPT_LEAD | {p}:{n}: {line[:220]}")
                        emitted += 1
                        if emitted >= 200:
                            return lines
            except OSError:
                pass
    return lines


def run(pkg: Path, reveal=False):
    os.umask(0o077)
    os.chdir(Path.home())
    if not BASE.is_dir() or BASE.is_symlink():
        raise SystemExit("审计目录不存在或为符号链接；停止。")

    out = fresh_output()
    report = []
    report.append("PeerJ 141707 | Label v1 Stage A full-source reconstruction")
    report.append("Status intent: evidence ledger + legacy-diff triage; NO FINAL LABELS; NO TRAINING")
    report.append("Local time: " + datetime.now().astimezone().isoformat())
    report.append("Output: " + str(out))
    report.append("")

    print("[1/6] 核验固定输入和既有候选包……", flush=True)
    pins = verify_pins()
    manifest = ensure_package(pkg)
    panel = load_panel(PINS["panel"][0])

    print("[2/6] 读取20,533条来源事件并重新识别机关/文书阶段/主体/报告年度……", flush=True)
    events = {}
    event_rows = []
    ecounts = Counter()
    for ev in read_jsonl(pkg / "event_evidence.jsonl"):
        raw = ev.get("raw", {})
        auth = authority_class(raw.get("Supervisor"), raw.get("Promulgator"))
        stage = document_stage(raw.get("FileName"), raw.get("PunishmentMeasure"), auth)
        subject = company_subject_status(raw, stage)
        yrs = extract_report_years(raw.get("Activity"))
        row = {
            "row_ref": ev.get("row_ref"),
            "source_row_number": ev.get("source_row_number"),
            "company_code": ev.get("normalized_company_code"),
            "violation_id": txt(raw.get("ViolationID")) or None,
            "company_name": txt(raw.get("CoFullName")) or None,
            "short_name": txt(raw.get("ShortName")) or None,
            "supervisor": txt(raw.get("Supervisor")) or None,
            "promulgator": txt(raw.get("Promulgator")) or None,
            "authority_class": auth,
            "document_stage": stage,
            "company_subject_status": subject,
            "violation_type_id_raw": txt(raw.get("ViolationTypeID")) or None,
            "violation_year_raw": txt(raw.get("ViolationYear")) or None,
            "punishment_type_id_raw": txt(raw.get("PunishmentTypeID")) or None,
            "file_name": txt(raw.get("FileName")) or None,
            "document_number": txt(raw.get("DocumentNumber")) or None,
            "declare_date": txt(raw.get("DeclareDate")) or None,
            "disposal_date": txt(raw.get("DisposalDate")) or None,
            "annual_report_years_text": yrs["annual_report_years"],
            "semiannual_report_years_text": yrs["semiannual_report_years"],
            "quarterly_report_years_text": yrs["quarterly_report_years"],
            "periodic_report_years_text": yrs["periodic_report_years"],
            "source_sha256": ev.get("source_sha256"),
            "evidence_level": "database_record_machine_classified_not_final_adjudication",
        }
        events[row["row_ref"]] = (row, raw)
        event_rows.append(row)
        ecounts["source_events"] += 1
        ecounts["authority__" + auth] += 1
        ecounts["stage__" + stage] += 1
        ecounts["subject__" + subject] += 1
    if len(events) != ecounts["source_events"]:
        raise ValueError("DUPLICATE_ROW_REF")

    print("[3/6] 对原类型—年度配对重新分流，并保留strict/loose两套证据状态……", flush=True)
    # record_candidates contains two scope copies; dedupe by underlying source pair.
    base_pairs = {}
    for rec in read_jsonl(pkg / "record_candidates.jsonl"):
        k = (rec.get("row_ref"), rec.get("group_index"), rec.get("violation_code"), rec.get("violation_year"))
        if k not in base_pairs:
            base_pairs[k] = rec
    pair_rows = []
    support = {"strict": defaultdict(list), "loose": defaultdict(list)}
    unresolved = {"strict": defaultdict(list), "loose": defaultdict(list)}
    pair_counts = {"strict": Counter(), "loose": Counter()}
    review_candidates = []

    for k, rec in base_pairs.items():
        row_ref, group_idx, code, year = k
        event, raw = events[row_ref]
        try:
            year = int(year) if year is not None else None
        except Exception:
            year = None
        ystatus = year_evidence_status(year, {
            "periodic_report_years": event["periodic_report_years_text"]
        })
        for scope in ("strict", "loose"):
            status = preanchor_status(scope, code, year, event["authority_class"], event["document_stage"], event["company_subject_status"], ystatus)
            key = (event["company_code"], year) if event["company_code"] and year is not None and 2010 <= year <= 2024 else None
            pr = {
                "row_ref": row_ref,
                "group_index": group_idx,
                "company_code": event["company_code"],
                "violation_id": event["violation_id"],
                "scope": scope,
                "violation_code": code,
                "raw_violation_year": year,
                "year_evidence_status": ystatus,
                "periodic_report_years_text": event["periodic_report_years_text"],
                "authority_class": event["authority_class"],
                "document_stage": event["document_stage"],
                "company_subject_status": event["company_subject_status"],
                "preanchor_status": status,
                "declare_date": event["declare_date"],
                "disposal_date": event["disposal_date"],
                "document_number": event["document_number"],
                "file_name": event["file_name"],
                "post_filing_status": "UNRESOLVED_NO_VERIFIED_ANNUAL_REPORT_FILING_ANCHOR",
                "final_label": None,
                "training_ready": False,
            }
            pair_rows.append(pr)
            pair_counts[scope][status] += 1
            if key:
                if status in {"A_text_year_supported_pre_anchor", "B_raw_year_only_pre_anchor"}:
                    support[scope][key].append((row_ref, status))
                elif status not in {"outside_type_scope", "nonfinal_document", "non_csrc_or_authority_unresolved"}:
                    unresolved[scope][key].append((row_ref, status))
            if status in {"year_conflict_requires_review", "company_subject_unresolved", "B_raw_year_only_pre_anchor"}:
                review_candidates.append(pr)

    print("[4/6] 形成51,675 firm-year的证据状态及legacy差异，不赋0/1新标签……", flush=True)
    fy_rows = []
    diff_rows = []
    diff_counts = {"strict": Counter(), "loose": Counter()}
    review_queue = []
    for (firm, year), old in sorted(panel.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        part = split_name(year)
        for scope, old_idx in (("strict", 1), ("loose", 2)):
            sup = support[scope].get((firm, year), [])
            unr = unresolved[scope].get((firm, year), [])
            if any(s == "A_text_year_supported_pre_anchor" for _, s in sup):
                state = "A_text_year_supported_pre_anchor"
            elif sup:
                state = "B_raw_year_only_pre_anchor"
            elif unr:
                state = "unresolved_pre_anchor"
            else:
                state = "no_machine_support_not_negative"
            legacy = old[old_idx]
            if legacy == 1 and state == "A_text_year_supported_pre_anchor":
                cat = "legacy_positive_text_supported_pre_anchor"
            elif legacy == 1 and state == "B_raw_year_only_pre_anchor":
                cat = "legacy_positive_raw_year_only_pre_anchor"
            elif legacy == 1 and state == "unresolved_pre_anchor":
                cat = "legacy_positive_unresolved"
            elif legacy == 1:
                cat = "legacy_positive_without_machine_support_NOT_adjudicated_error"
            elif legacy == 0 and state in {"A_text_year_supported_pre_anchor", "B_raw_year_only_pre_anchor"}:
                cat = "legacy_zero_with_new_formal_preanchor_support"
            elif legacy == 0 and state == "unresolved_pre_anchor":
                cat = "legacy_zero_unresolved"
            else:
                cat = "legacy_zero_no_machine_support_not_negative"
            diff_counts[scope][cat] += 1
            fy = {
                "company_code": firm,
                "fiscal_year": year,
                "partition": part,
                "scope": scope,
                "preanchor_evidence_state": state,
                "support_event_count": len(sup),
                "unresolved_event_count": len(unr),
                "support_row_refs": [r for r, _ in sup],
                "unresolved_row_refs": [r for r, _ in unr],
                "legacy_label": legacy,
                "legacy_comparison_category": cat,
                "post_filing_status": "UNRESOLVED_NO_VERIFIED_ANNUAL_REPORT_FILING_ANCHOR",
                "final_label": None,
                "training_ready": False,
            }
            fy_rows.append(fy)
            diff_rows.append({k: fy[k] for k in (
                "company_code", "fiscal_year", "partition", "scope", "legacy_label",
                "preanchor_evidence_state", "support_event_count", "unresolved_event_count",
                "legacy_comparison_category", "post_filing_status")})
            if cat in {
                "legacy_positive_unresolved",
                "legacy_positive_without_machine_support_NOT_adjudicated_error",
                "legacy_zero_with_new_formal_preanchor_support",
            }:
                review_queue.append(diff_rows[-1])

    if len(fy_rows) != 51675 * 2:
        raise AssertionError("FIRM_YEAR_SCOPE_ROWCOUNT_FAILED")

    print("[5/6] 写出本机私有证据文件，并查找年报披露锚点线索……", flush=True)
    write_parquet(out / "event_ledger_v1_stageA.parquet", event_rows)
    write_parquet(out / "pair_ledger_v1_stageA.parquet", pair_rows)
    write_parquet(out / "firm_year_support_v1_stageA.parquet", fy_rows)

    # CSVs are private/local. Prefix stock code to preserve leading zeros in spreadsheet apps.
    with (out / "legacy_diff_stageA_LOCAL_ONLY.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(diff_rows[0])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in diff_rows:
            x = dict(r); x["company_code"] = "'" + x["company_code"]
            w.writerow(x)
    with (out / "review_queue_stageA_LOCAL_ONLY.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(review_queue[0]) if review_queue else ["company_code", "fiscal_year", "scope"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in review_queue:
            x = dict(r); x["company_code"] = "'" + x["company_code"]
            w.writerow(x)

    anchors = anchor_inventory(PROJECT)
    (out / "anchor_source_inventory_LOCAL_ONLY.txt").write_text("\n".join(anchors) + "\n", encoding="utf-8")

    protocol = {
        "version": VERSION,
        "purpose": "Stage A evidence reconstruction only; no final labels/training",
        "source_package": str(pkg),
        "cutoff": CUTOFF,
        "strict_codes": sorted(STRICT),
        "loose_codes": sorted(LOOSE),
        "authority_rule": "CSRC HQ and CSRC branch are eligible authority classes; exchanges separated",
        "document_stage_rule": "prior notices and regulatory/exchange measures are separated from final administrative penalties",
        "subject_rule": "company full/short name, dedicated company punishment field, or explicit 对你公司 cue; unresolved otherwise",
        "year_rule": "raw aligned ViolationYear retained; explicit periodic-report years extracted only as consistency evidence; conflicts are never auto-corrected",
        "post_filing_rule": "not applied in Stage A; requires verified annual-report filing anchor and earliest qualifying public-sanction date",
        "negative_rule": "absence of machine support is never a negative label in Stage A",
        "legacy_rule": "legacy labels used only for comparison/triage, never as eligibility input",
    }
    (out / "protocol_v1_stageA.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report.append("A. INPUT / SAFETY")
    report.append("candidate_package=" + str(pkg))
    report.append("pinned_inputs_verified=" + dumps({k: v["sha256"] for k, v in pins.items()}))
    report.append("source_files_modified=false")
    report.append("training_or_prediction_run=false")
    report.append("final_nonnull_labels_written=0")
    report.append("")
    report.append("B. EVENT CLASSIFICATION COUNTS")
    report.append(dumps(dict(ecounts)))
    report.append("")
    report.append("C. TYPE-YEAR PAIR PRE-ANCHOR STATUS")
    report.append("strict=" + dumps(dict(pair_counts["strict"])))
    report.append("loose=" + dumps(dict(pair_counts["loose"])))
    report.append("")
    report.append("D. LEGACY vs STAGE-A SUPPORT (NOT ERROR ADJUDICATION)")
    report.append("strict=" + dumps(dict(diff_counts["strict"])))
    report.append("loose=" + dumps(dict(diff_counts["loose"])))
    report.append("")
    report.append("E. REVIEW WORKLOAD")
    report.append(f"review_queue_firm_year_scope_rows={len(review_queue)}")
    report.append("Queue is priority triage, not an estimated error rate.")
    report.append("")
    report.append("F. BLOCKING ITEM BEFORE FINAL LABELS")
    report.append("verified annual-report filing anchors + earliest qualifying public-sanction date are still required for post-filing status.")
    report.append("No final 0/1 label is authorized from this Stage-A output.")
    report.append("")
    report.append("Upload only label_v1_stageA_report.txt. Other outputs contain licensed/company-level material and stay local.")

    (out / "label_v1_stageA_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    files = {}
    for p in sorted(out.iterdir()):
        if p.is_file():
            files[p.name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    mani = {
        "version": VERSION,
        "completed": True,
        "created_at": datetime.now().astimezone().isoformat(),
        "candidate_package_manifest_sha256": sha256(pkg / "manifest.json"),
        "input_files": pins,
        "outputs": files,
        "final_labels_authorized": False,
        "training_authorized": False,
    }
    (out / "manifest_v1_stageA.json").write_text(json.dumps(mani, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[6/6] 完成 Stage A；没有生成最终标签，也没有训练。", flush=True)
    print("STATUS: LABEL_V1_STAGE_A_COMPLETE_POST_FILING_ANCHORS_REQUIRED")
    print("请只上传：")
    print(out / "label_v1_stageA_report.txt")
    if reveal:
        print("本地输出目录：")
        print(out)
    return out


def main():
    global START
    START = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-a", action="store_true", help="Run Stage A evidence reconstruction; no final labels")
    ap.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    ap.add_argument("--reveal", action="store_true")
    args = ap.parse_args()
    if not args.stage_a:
        ap.error("--stage-a is required")
    run(args.package, reveal=args.reveal)


if __name__ == "__main__":
    main()
