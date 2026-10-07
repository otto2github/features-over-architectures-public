#!/usr/bin/env python3
"""No-training RPT identity and annual-incidence audit; Python standard library."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import sys
import tempfile
import time
import unicodedata
import xml.etree.ElementTree as ET
import zipfile

PROJECT = Path('/path/to/private_source_project')
VERSION = '1.0'
VOLUMES = ['RPT_Operation.xlsx', 'RPT_Operation1.xlsx', 'RPT_Operation2.xlsx']
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
FIELDS = ['Stkcd', 'Reptdt', 'Repart', 'RalatedPartyID', 'Relation', 'Direction',
          'Repat', 'Kind', 'Isam', 'Pannrsm', 'Isgplo', 'Ifafcprf']
SENTINELS = {'', 'nan', 'none', 'null', 'n/a', 'na', '-', '--', '0', '0.0', '-1', '-1.0'}
GENERIC_NAMES = {'其他', '其它', '其他关联方', '其它关联方', '关联方', '关联公司',
                 '关联企业', '其他关联企业', '其他关联公司', '不适用', '未知', '无'}
RETURN_FILES = ['AUDIT_SUMMARY.json', 'ANNUAL_STRUCTURE.csv', 'SOURCE_IDENTITIES.json',
                'FIELD_METADATA.json', 'OLD_E7_BUILDER_REDACTED.py', 'README_RETURN.txt',
                'RUNNER_IDENTITIES.json']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def within(path, root):
    return Path(path).is_relative_to(root)


def checked_output(path, boundary):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError('OUTPUT_SYMLINK_COMPONENT')
    resolved = path.resolve()
    if resolved == boundary or not within(resolved, boundary):
        raise ValueError('OUTPUT_OUTSIDE_WRITE_BOUNDARY')
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise ValueError('OUTPUT_EXISTS_NOT_EMPTY')
    return resolved


def first_sheet(book):
    wb = ET.fromstring(book.read('xl/workbook.xml'))
    props = wb.find(NS + 'workbookPr')
    date1904 = props is not None and props.attrib.get('date1904', '').lower() in {'1', 'true'}
    sheets = wb.find(NS + 'sheets')
    if sheets is None or not len(sheets):
        raise ValueError('NO_WORKSHEET')
    rid = sheets[0].attrib.get(REL + 'id')
    rs = ET.fromstring(book.read('xl/_rels/workbook.xml.rels'))
    import posixpath
    for r in rs:
        if r.attrib.get('Id') == rid:
            target = r.attrib.get('Target', '')
            member = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
            if member.startswith('../') or member not in book.namelist():
                raise ValueError('INVALID_WORKSHEET_TARGET')
            return member, date1904, len(sheets)
    raise ValueError('WORKSHEET_RELATIONSHIP_MISSING')


class Workbook:
    """Stream sheet rows; disk-backed shared strings stay in a private temp DB."""
    def __init__(self, path, db, volume):
        self.book = zipfile.ZipFile(path)
        self.db = db
        self.volume = volume
        self.member, self.date1904, self.sheet_count = first_sheet(self.book)
        if self.sheet_count != 1:
            self.book.close()
            raise ValueError('MULTIPLE_SHEETS_REQUIRE_EXPLICIT_SELECTION')
        self.db.execute('CREATE TABLE IF NOT EXISTS strings (v INTEGER, i INTEGER, t TEXT, PRIMARY KEY(v,i)) WITHOUT ROWID')
        self.nstrings = 0
        if 'xl/sharedStrings.xml' in self.book.namelist():
            batch = []
            with self.book.open('xl/sharedStrings.xml') as stream:
                root = None
                for event, element in ET.iterparse(stream, events=('start', 'end')):
                    if root is None:
                        root = element
                    if event == 'end' and element.tag == NS + 'si':
                        value = ''.join(t.text or '' for t in element.iter(NS + 't'))
                        batch.append((volume, self.nstrings, value))
                        self.nstrings += 1
                        if len(batch) >= 5000:
                            db.executemany('INSERT INTO strings VALUES (?,?,?)', batch)
                            batch.clear()
                        root.clear()
            if batch:
                db.executemany('INSERT INTO strings VALUES (?,?,?)', batch)
            db.commit()

    @lru_cache(maxsize=65536)
    def string(self, index):
        row = self.db.execute('SELECT t FROM strings WHERE v=? AND i=?', (self.volume, index)).fetchone()
        if row is None:
            raise ValueError('INVALID_SHARED_STRING_INDEX')
        return row[0]

    def rows(self):
        with self.book.open(self.member) as stream:
            sheet_data = None
            sequence = 0
            for event, element in ET.iterparse(stream, events=('start', 'end')):
                if event == 'start' and element.tag == NS + 'sheetData':
                    sheet_data = element
                if event == 'end' and element.tag == NS + 'row':
                    sequence += 1
                    physical = int(element.attrib.get('r', sequence))
                    cells = {}
                    for c in element.findall(NS + 'c'):
                        ref = c.attrib.get('r', '')
                        match = re.match(r'([A-Z]+)', ref)
                        if not match:
                            raise ValueError('CELL_COLUMN_REFERENCE_MISSING')
                        column = 0
                        for ch in match.group(1):
                            column = column * 26 + ord(ch) - 64
                        kind = c.attrib.get('t', 'n')
                        v = c.find(NS + 'v')
                        if kind == 's':
                            value = self.string(int(v.text)) if v is not None and v.text else ''
                        elif kind == 'inlineStr':
                            value = ''.join(t.text or '' for t in c.iter(NS + 't'))
                        else:
                            value = v.text or '' if v is not None else ''
                        cells[column - 1] = (value, kind)
                    yield physical, cells
                    if sheet_data is not None:
                        sheet_data.clear()

    def close(self):
        self.string.cache_clear()
        self.db.execute('DELETE FROM strings WHERE v=?', (self.volume,))
        self.db.commit()
        self.book.close()


def clean(value):
    return unicodedata.normalize('NFKC', str(value)).strip()


def issuer(value):
    value = clean(value)
    match = re.fullmatch(r'(\d{1,6})(?:\.0+|\.(?:SH|SZ|BJ))?', value, flags=re.I)
    if match and int(match.group(1)) > 0:
        return match.group(1).zfill(6)
    return None


def report_date(value, kind, date1904):
    value = clean(value)
    if not value:
        return None, 'missing'
    if re.fullmatch(r'(?:19|20)\d{2}(?:\.0+)?', value):
        return date(int(Decimal(value)), 1, 1), 'year_only'
    if re.fullmatch(r'(?:19|20)\d{6}', value):
        try:
            return date(int(value[:4]), int(value[4:6]), int(value[6:8])), 'yyyymmdd'
        except ValueError:
            return None, 'invalid'
    m = re.match(r'^((?:19|20)\d{2})[-/年](\d{1,2})[-/月](\d{1,2})(?:日)?(?:$|[T\s])', value)
    if m:
        try:
            return date(*(int(x) for x in m.groups())), 'calendar_date'
        except ValueError:
            return None, 'invalid'
    if kind == 'n':
        try:
            serial = Decimal(value)
            if Decimal(1) <= serial <= Decimal(80000):
                epoch = date(1904, 1, 1) if date1904 else date(1899, 12, 30)
                result = epoch + timedelta(days=int(serial))
                if 1900 <= result.year <= 2100:
                    return result, 'excel_serial'
        except (InvalidOperation, ValueError, OverflowError):
            pass
    return None, 'invalid'


def party_key(value, kind):
    value = clean(value)
    precision_flag = False
    if kind == 'n' and value:
        try:
            number = Decimal(value)
            if not number.is_finite():
                return '', False
            if number == number.to_integral_value():
                precision_flag = abs(number) >= Decimal(10) ** 15
                value = format(number, 'f').split('.')[0]
        except InvalidOperation:
            return value, True
    return value, precision_flag


def norm_name(value):
    # Diagnostic only: no fuzzy matching, punctuation/suffix removal, or legal-entity join.
    return re.sub(r'\s+', '', clean(value)).casefold()


def init_db(db):
    db.execute('PRAGMA journal_mode=OFF')
    db.execute('PRAGMA synchronous=OFF')
    db.execute('PRAGMA temp_store=MEMORY')
    db.execute('PRAGMA cache_size=-32768')
    db.executescript('''
      CREATE TABLE edges (year INTEGER, firm TEXT, party TEXT, nrows INTEGER, volumes INTEGER,
                          PRIMARY KEY(year,firm,party)) WITHOUT ROWID;
      CREATE TABLE names (year INTEGER, firm TEXT, party TEXT, nrows INTEGER,
                          PRIMARY KEY(year,firm,party)) WITHOUT ROWID;
      CREATE TABLE identities (party TEXT, firm TEXT, raw_name TEXT, norm_name TEXT,
                               minyear INTEGER, maxyear INTEGER, volumes INTEGER, nrows INTEGER,
                               PRIMARY KEY(party,firm,raw_name,norm_name)) WITHOUT ROWID;
    ''')


def scalar(db, query):
    row = db.execute(query).fetchone()
    return int(row[0] or 0)


def read_volume(path, db, volume, secret):
    book = Workbook(path, db, volume)
    counts = Counter()
    yearly = {}
    precision = Counter()
    date_md = Counter()
    id_types = Counter()
    min_date = max_date = None
    header = None
    known_columns = []
    edge_batch, name_batch, identity_batch = [], [], []

    @lru_cache(maxsize=131072)
    def digest(namespace, value):
        return hmac.new(secret, (namespace + '\0' + value).encode('utf-8'), hashlib.sha256).hexdigest()

    def flush():
        db.executemany('''INSERT INTO edges VALUES (?,?,?,?,?)
            ON CONFLICT(year,firm,party) DO UPDATE SET nrows=nrows+excluded.nrows,
            volumes=volumes | excluded.volumes''', edge_batch)
        db.executemany('''INSERT INTO names VALUES (?,?,?,?)
            ON CONFLICT(year,firm,party) DO UPDATE SET nrows=nrows+excluded.nrows''', name_batch)
        db.executemany('''INSERT INTO identities VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(party,firm,raw_name,norm_name) DO UPDATE SET
            minyear=MIN(minyear,excluded.minyear), maxyear=MAX(maxyear,excluded.maxyear),
            volumes=volumes | excluded.volumes, nrows=nrows+excluded.nrows''', identity_batch)
        edge_batch.clear(); name_batch.clear(); identity_batch.clear()
        db.commit()

    try:
        for physical, cells in book.rows():
            if physical == 1:
                header = {}
                for index, (value, _) in cells.items():
                    key = clean(value).lstrip('\ufeff')
                    if key == 'RelatedPartyID':
                        key = 'RalatedPartyID'
                    if key in FIELDS:
                        if key in header:
                            raise ValueError('DUPLICATE_REQUIRED_COLUMN')
                        header[key] = index
                if not {'Stkcd', 'Reptdt', 'Repart', 'RalatedPartyID'}.issubset(header):
                    raise ValueError('REQUIRED_RPT_COLUMNS_MISSING')
                known_columns = sorted(header)
                continue
            if header is None:
                raise ValueError('FIRST_PHYSICAL_ROW_HEADER_MISSING')
            def cell(field):
                return cells.get(header[field], ('', 'n'))
            firm = issuer(cell('Stkcd')[0])
            dvalue, dkind = cell('Reptdt')
            d, date_precision = report_date(dvalue, dkind, book.date1904)
            if physical in {2, 3}:
                if firm is not None and d is not None:
                    raise ValueError('PRESUMED_METADATA_ROW_LOOKS_LIKE_RECORD')
                counts['metadata_rows_skipped'] += 1
                continue
            if not any(v for v, _ in cells.values()):
                counts['empty_rows'] += 1
                continue
            counts['data_rows'] += 1
            if counts['data_rows'] % 100000 == 0:
                print('Volume ' + str(volume + 1) + ': ' + str(counts['data_rows']) + ' rows read', flush=True)
            precision[date_precision] += 1
            if firm is None:
                counts['invalid_issuer_rows'] += 1
            if d is None:
                counts['invalid_or_missing_date_rows'] += 1
            if firm is None or d is None:
                continue
            counts['valid_issuer_date_rows'] += 1
            min_date = min(min_date, d) if min_date else d
            max_date = max(max_date, d) if max_date else d
            if date_precision != 'year_only':
                date_md[d.strftime('%m-%d')] += 1
            year = d.year
            yc = yearly.setdefault(year, Counter())
            yc['valid_issuer_date_rows'] += 1
            pid_value, pid_type = cell('RalatedPartyID')
            pid, unsafe_numeric = party_key(pid_value, pid_type)
            id_types[pid_type] += 1
            if unsafe_numeric:
                counts['numeric_id_precision_risk_rows'] += 1
                yc['numeric_id_precision_risk_rows'] += 1
            if pid_type in {'s', 'inlineStr', 'str'} and re.fullmatch(r'0\d+', pid):
                counts['text_ids_with_leading_zero_rows'] += 1
            raw_name = str(cell('Repart')[0]).strip()
            normalized_name = norm_name(raw_name)
            missing_name = normalized_name.lower() in SENTINELS
            if missing_name:
                counts['missing_name_rows'] += 1
                yc['missing_name_rows'] += 1
            generic = normalized_name in GENERIC_NAMES
            if generic:
                counts['generic_name_candidate_rows'] += 1
                yc['generic_name_candidate_rows'] += 1
            firm_hash = digest('firm', firm)
            if not missing_name:
                name_batch.append((year, firm_hash, digest('name_norm', normalized_name), 1))
            if pid.lower() in SENTINELS:
                counts['missing_or_sentinel_id_rows'] += 1
                yc['missing_or_sentinel_id_rows'] += 1
            else:
                counts['candidate_id_rows'] += 1
                yc['candidate_id_rows'] += 1
                party_hash = digest('party', pid)
                edge_batch.append((year, firm_hash, party_hash, 1, 1 << volume))
                identity_batch.append((party_hash, firm_hash,
                    '' if missing_name else digest('name_raw', raw_name),
                    '' if missing_name else digest('name_norm', normalized_name),
                    year, year, 1 << volume, 1))
            if len(edge_batch) + len(name_batch) >= 10000:
                flush()
        flush()
    finally:
        book.close()
        digest.cache_clear()
    return {
        'path': str(path), 'volume_index': volume + 1, 'counts': dict(counts),
        'known_columns': known_columns, 'date_precision_counts': dict(precision),
        'report_month_day_counts': dict(sorted(date_md.items())),
        'id_excel_cell_type_counts': dict(id_types),
        'min_report_date': min_date.isoformat() if min_date else None,
        'max_report_date': max_date.isoformat() if max_date else None,
        'excel_date1904': book.date1904,
        'selected_worksheet': 'first_and_only_worksheet',
        'year_counts': {str(y): dict(c) for y, c in sorted(yearly.items())},
    }


def annual_structure(db, table):
    results = []
    for (year,) in db.execute('SELECT DISTINCT year FROM ' + table + ' ORDER BY year').fetchall():
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                parent[b] = a
        total_edges = 0
        party_degrees = []
        shared_firms = set()
        previous = None
        current = []
        def close_party():
            if not current:
                return
            party_degrees.append(len(current))
            if len(current) >= 2:
                shared_firms.update(current)
                for firm in current[1:]:
                    union(current[0], firm)
        for party, firm in db.execute('SELECT party,firm FROM ' + table + ' WHERE year=? ORDER BY party,firm', (year,)):
            total_edges += 1
            find(firm)
            if party != previous:
                close_party()
                current = []
                previous = party
            current.append(firm)
        close_party()
        components = Counter(find(firm) for firm in parent)
        sizes = sorted(components.values(), reverse=True)
        result = {
            'year': int(year), 'unique_issuer_party_year_edges': total_edges,
            'unique_issuers': len(parent), 'unique_party_keys': len(party_degrees),
            'party_keys_degree_1': sum(d == 1 for d in party_degrees),
            'party_keys_degree_ge_2': sum(d >= 2 for d in party_degrees),
            'party_keys_degree_ge_5': sum(d >= 5 for d in party_degrees),
            'party_keys_degree_ge_10': sum(d >= 10 for d in party_degrees),
            'party_keys_degree_ge_50': sum(d >= 50 for d in party_degrees),
            'max_party_distinct_issuer_degree': max(party_degrees, default=0),
            'issuers_with_shared_party': len(shared_firms),
            'issuer_shared_party_fraction': len(shared_firms) / len(parent) if parent else None,
            'incidence_edges_to_shared_parties': sum(d for d in party_degrees if d >= 2),
            'rpt_only_issuer_component_count': len(sizes),
            'rpt_only_largest_issuer_component': sizes[0] if sizes else 0,
            'largest_component_fraction': sizes[0] / len(parent) if parent else None,
            'top_20_party_degrees_no_identifiers': sorted(party_degrees, reverse=True)[:20],
            'primary_graph_year_2010_2022': 2010 <= year <= 2022,
        }
        results.append(result)
    return results


def identity_stats(db):
    return {
        'unique_candidate_ids': scalar(db, 'SELECT COUNT(DISTINCT party) FROM identities'),
        'unique_candidate_issuers': scalar(db, 'SELECT COUNT(DISTINCT firm) FROM identities'),
        'ids_seen_at_multiple_issuers': scalar(db, 'SELECT COUNT(*) FROM (SELECT party FROM identities GROUP BY party HAVING COUNT(DISTINCT firm)>1)'),
        'ids_with_multiple_raw_names': scalar(db, "SELECT COUNT(*) FROM (SELECT party FROM identities WHERE raw_name<>'' GROUP BY party HAVING COUNT(DISTINCT raw_name)>1)"),
        'ids_with_multiple_normalized_names': scalar(db, "SELECT COUNT(*) FROM (SELECT party FROM identities WHERE norm_name<>'' GROUP BY party HAVING COUNT(DISTINCT norm_name)>1)"),
        'issuer_id_pairs_with_multiple_normalized_names': scalar(db, "SELECT COUNT(*) FROM (SELECT party,firm FROM identities WHERE norm_name<>'' GROUP BY party,firm HAVING COUNT(DISTINCT norm_name)>1)"),
        'ids_with_name_conflicts_only_across_issuers': scalar(db, """SELECT COUNT(*) FROM
            (SELECT party FROM identities WHERE norm_name<>'' GROUP BY party HAVING COUNT(DISTINCT norm_name)>1)
            WHERE party NOT IN (SELECT party FROM identities WHERE norm_name<>'' GROUP BY party,firm HAVING COUNT(DISTINCT norm_name)>1)"""),
        'normalized_names_associated_with_multiple_ids': scalar(db, "SELECT COUNT(*) FROM (SELECT norm_name FROM identities WHERE norm_name<>'' GROUP BY norm_name HAVING COUNT(DISTINCT party)>1)"),
        'ids_spanning_multiple_report_years': scalar(db, 'SELECT COUNT(*) FROM (SELECT party FROM identities GROUP BY party HAVING MIN(minyear)<MAX(maxyear))'),
        'ids_spanning_multiple_volumes': scalar(db, 'SELECT COUNT(*) FROM (SELECT party FROM identities GROUP BY party HAVING MAX(volumes) NOT IN (1,2,4) OR MIN(volumes)<>MAX(volumes))'),
        'unique_incidence_edges_all_years': scalar(db, 'SELECT COUNT(*) FROM edges'),
        'incidence_edges_repeated_across_rows': scalar(db, 'SELECT COUNT(*) FROM edges WHERE nrows>1'),
        'additional_rows_beyond_unique_incidence_edges': scalar(db, 'SELECT SUM(nrows-1) FROM edges'),
        'incidence_edges_present_in_multiple_volumes': scalar(db, 'SELECT COUNT(*) FROM edges WHERE volumes NOT IN (1,2,4)'),
        'max_rows_per_incidence_edge': scalar(db, 'SELECT MAX(nrows) FROM edges'),
        'notes': [
            'An incidence-edge repeat can represent distinct transactions or report periods; it is not proof of duplicate transactions.',
            'ID/name disagreement can reflect collisions, name changes, aliases, or provider encoding. It requires source-semantic review.',
            'Consistent names alone cannot prove a globally stable provider ID.',
        ],
    }


def descriptors(path):
    if not path.exists():
        return {'status': 'DES_FILE_NOT_FOUND'}
    raw = path.read_bytes()
    if len(raw) > 1024 * 1024:
        return {'status': 'DES_FILE_SIZE_LIMIT'}
    text = None
    for encoding in ('utf-8-sig', 'gb18030', 'utf-16'):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeError:
            continue
    if text is None:
        return {'status': 'DES_ENCODING_UNSUPPORTED'}
    # Return field descriptions, not full export/query metadata or examples.
    lines = text.splitlines()
    excerpts = {}
    for field in ['RalatedPartyID', 'RelatedPartyID', 'Repart', 'Reptdt', 'Relation', 'Direction', 'Repat']:
        snippets = []
        for i, line in enumerate(lines):
            if re.search(r'\b' + re.escape(field) + r'\b', line):
                if re.search(r'示例|样例|例如|例子|Example|Sample', line, flags=re.I):
                    continue
                selected = [line]
                for following in lines[i + 1:i + 4]:
                    if any(re.search(r'\b' + re.escape(f) + r'\b', following) for f in FIELDS):
                        break
                    if not following.strip():
                        break
                    if re.search(r'示例|样例|例如|例子|Example|Sample', following, flags=re.I):
                        break
                    if re.match(r'^\s*(?:字段说明|字段解释|说明|定义|类型|长度|单位|描述|Field description|Description|Type|Length|Unit)\s*[:：]', following, flags=re.I):
                        selected.append(following)
                snippets.append('\n'.join(selected)[:1500])
        if snippets:
            excerpts[field] = snippets[:4]
    return {'status': 'FIELD_DESCRIPTION_EXCERPTS_ONLY', 'field_excerpts': excerpts,
            'scope_note': 'Provider field descriptions require human interpretation; no automatic global-ID assertion.'}


def redact_builder(path):
    if not path.exists():
        return '# OLD_E7_BUILDER_NOT_FOUND\n', {'status': 'NOT_FOUND'}
    if path.stat().st_size > 4 * 1024 * 1024:
        return '# OLD_E7_BUILDER_SIZE_LIMIT\n', {'status': 'SIZE_LIMIT'}
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    allowed = set(FIELDS + ['RelatedPartyID', 'firm_id', 'year', 'src', 'dst', 'src_id', 'dst_id',
        'edge_type', 'relation', 'relation_type', 'id', 'type', 'weight', 'amount', 'node_id',
        'company', 'institution', 'person', 'utf-8', 'snappy', 'E7', 'I:', 'P:', 'C:', 'RPT:',
        '', ' ', '\\s+', '\\W+', '^\\d{6}$', '贷款', '借款', '拆借', '担保', '保证'])
    class Redact(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                value = node.value
                if value not in allowed and not re.fullmatch(r'E7_[A-Z_]+', value) and not re.fullmatch(r'RPT_Operation\d*\.xlsx', value):
                    return ast.copy_location(ast.Constant(value='<STRING_REDACTED>'), node)
            elif isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                if abs(node.value) > 2100:
                    return ast.copy_location(ast.Constant(value='<LARGE_NUMBER_REDACTED>'), node)
            return node
    tree = Redact().visit(tree)
    ast.fix_missing_locations(tree)
    return '# STATIC REVIEW COPY ONLY; NEVER EXECUTE\n# Comments/docstrings and unapproved string literals withheld.\n' + ast.unparse(tree) + '\n', {
        'status': 'AST_REDACTED_STATIC_REVIEW_ONLY',
        'known_field_tokens': {f: f in path.read_text(encoding='utf-8-sig') for f in FIELDS},
        'execution_proof': False,
    }


def run(args):
    source_root = Path(args.source_root).absolute().resolve(strict=True)
    if args.local_test:
        if PROJECT.exists() or Path('/path/to/private_workspace').exists():
            raise ValueError('LOCAL_TEST_DISABLED_ON_PRODUCTION_SERVER')
        boundary = source_root
    else:
        if source_root != PROJECT or PROJECT.is_symlink():
            raise ValueError('PRODUCTION_SOURCE_ROOT_NOT_CANONICAL_PROJECT')
        boundary = PROJECT
    output = checked_output(args.output_dir, boundary)
    sources = [source_root / 'data/raw/csmar' / name for name in VOLUMES]
    for path in sources:
        if not path.is_file() or not within(path.resolve(), source_root):
            raise ValueError('RPT_VOLUME_MISSING_OR_OUTSIDE_SOURCE_ROOT')
    output.mkdir(parents=True)
    start = time.time()
    identities = []
    aux = [source_root / 'data/raw/csmar/RPT_Operation[DES][xlsx].txt',
           source_root / 'scripts/p1_v1_integration/scripts/build_kg_e7_trade.py',
           source_root / 'scripts/p1_v1_integration/scripts/integrate_rpt_operation_features.py',
           source_root / 'scripts/p1_v1_integration/scripts/_v1_common.py']
    for path in sources + aux:
        if path.exists():
            if not within(path.resolve(), source_root):
                raise ValueError('AUXILIARY_SOURCE_OUTSIDE_ROOT')
            print('Hashing source: ' + path.name, flush=True)
            identities.append({'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': sha(path)})
    write_json(output / 'SOURCE_IDENTITIES.json', {'sources': identities,
        'original_acquisition_dates_established': False,
        'source_mtimes_not_used_as_acquisition_dates': True})
    write_json(output / 'FIELD_METADATA.json', descriptors(aux[0]))
    builder, builder_metadata = redact_builder(aux[1])
    (output / 'OLD_E7_BUILDER_REDACTED.py').write_text(builder, encoding='utf-8')

    per_volume = []
    with tempfile.TemporaryDirectory(prefix='private_audit_', dir=output) as private:
        db = sqlite3.connect(str(Path(private) / 'private_cache.sqlite'))
        try:
            init_db(db)
            secret = secrets.token_bytes(32)
            for index, path in enumerate(sources):
                print('Reading volume ' + str(index + 1) + ' of 3', flush=True)
                per_volume.append(read_volume(path, db, index, secret))
            stats = identity_stats(db)
            by_id = annual_structure(db, 'edges')
            by_name = annual_structure(db, 'names')
        finally:
            db.close()
    total = Counter()
    for volume in per_volume:
        total.update(volume['counts'])
    source_check = {row['path']: sha(Path(row['path'])) == row['sha256'] for row in identities}
    if not all(source_check.values()):
        raise ValueError('SOURCE_CHANGED_DURING_AUDIT')
    summary = {
        'status': 'STRUCTURAL_AUDIT_COMPLETE_REVIEW_REQUIRED', 'version': VERSION,
        'training_started': False, 'labels_or_scores_read': False,
        'global_party_id_semantics_verified': False,
        'source_recheck_match': source_check,
        'total_counts': dict(total), 'per_volume': per_volume,
        'identity_diagnostics': stats,
        'annual_id_key_incidence_diagnostics': by_id,
        'annual_exact_normalized_name_diagnostics_not_a_training_graph': by_name,
        'old_builder_static_evidence': builder_metadata,
        'required_review': [
            'Read the provider description and resolve global vs issuer-local party ID semantics.',
            'Inspect cross-issuer ID/name disagreement, sentinel candidates and numeric precision risk.',
            'Inspect same-year sharing and concentration before choosing a graph construction.',
            'Audit source/company-index alignment and eligible-company coverage on the frozen baseline separately.',
            'Report-year incidence does not establish historical disclosure availability.',
            'The normalized-name diagnostic is not a validated legal-entity crosswalk or an alternative selected training graph.',
            'Degree-one zero-feature parties can change model normalization and messages; they are not inert.',
        ],
        'private_database_retained': False, 'raw_ids_names_rows_exported': False,
        'elapsed_seconds': round(time.time() - start, 3),
    }
    write_json(output / 'AUDIT_SUMMARY.json', summary)
    import csv
    with (output / 'ANNUAL_STRUCTURE.csv').open('w', encoding='utf-8-sig', newline='') as f:
        fields = ['key_basis'] + [k for k in (by_id or by_name or [{}])[0] if k != 'top_20_party_degrees_no_identifiers']
        if len(fields) == 1:
            fields = ['key_basis', 'year']
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader()
        for basis, rows in [('provider_id_candidate', by_id), ('normalized_name_diagnostic_only', by_name)]:
            for row in rows:
                writer.writerow({'key_basis': basis, **row})
    package = Path(__file__).resolve().parent
    write_json(output / 'RUNNER_IDENTITIES.json', {
        p.name: {'sha256': sha(p), 'size_bytes': p.stat().st_size}
        for p in [package / 'e7_structural_audit.py', package / 'run_structural_audit.sh',
                  package / 'STRUCTURAL_AUDIT_PROTOCOL_v1.md', package / 'README_RUN_TENCENT_CN.md']
        if p.is_file() and not p.is_symlink()})
    (output / 'README_RETURN.txt').write_text(
        'Private PeerJ 141707 RPT structural audit. No training was performed.\n'
        'Status STRUCTURAL_AUDIT_COMPLETE_REVIEW_REQUIRED is not a training gate PASS.\n'
        'The return contains aggregate counts, paths/hashes, field descriptions and an AST-redacted old builder.\n'
        'No source rows, company/party IDs, party names, labels or scores are exported.\n'
        'Keep this return private; do not publish it as a submission/reproducibility package.\n', encoding='utf-8')
    with zipfile.ZipFile(output / 'E7_STRUCTURAL_RETURN.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name in RETURN_FILES:
            z.write(output / name, arcname=name)
    print('STRUCTURAL_AUDIT_COMPLETE_REVIEW_REQUIRED', flush=True)
    print('Return ZIP: ' + str(output / 'E7_STRUCTURAL_RETURN.zip'), flush=True)
    return output / 'E7_STRUCTURAL_RETURN.zip'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', default=str(PROJECT))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--local-test', action='store_true')
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        raise ValueError('PYTHON_3_10_OR_NEWER_REQUIRED')
    return run(args)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        message = str(error)
        status = message if isinstance(error, ValueError) and re.fullmatch(r'[A-Z0-9_]+', message) else 'AUDIT_ERROR'
        print(json.dumps({'status': status, 'error_type': type(error).__name__}), file=sys.stderr)
        raise SystemExit(1)
