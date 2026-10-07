#!/usr/bin/env python3
"""Focused, label-free RPT identity diagnostics; no graph export or training."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
import sqlite3
import sys
import tempfile
import time
import zipfile

sys.dont_write_bytecode = True
import e7_xlsx_reader as reader

VERSION = '1.0'
KIT = Path(__file__).resolve().parent
PROJECT = Path('/path/to/private_source_project')
RETURN_FILES = ['FOCUS_SUMMARY.json', 'ANNUAL_IDENTITY.csv', 'ANNUAL_PROFILES.csv',
                'SOURCE_IDENTITIES.json', 'RUNNER_IDENTITIES.json', 'README_RETURN.txt']
RUNNER_FILES = ['e7_identity_focus.py', 'e7_xlsx_reader.py', 'run_identity_focus.sh',
                'IDENTITY_FOCUS_PROTOCOL_v1.md', 'README_RUN_TENCENT_CN.md',
                'PRIOR_AUDIT_REFERENCE.json', 'STAGE_II_ASSESSMENT_CN.md',
                'DEVELOPMENT_VALIDATION.json', 'PACKAGE_SHA256.json']
# Operational exclusions for diagnostics, never a claim of legal-entity validation.
EXTRA_MISSING_NAMES = {'未提供', '未披露', '不详', '未知关联方', '未列示', '未明确',
                       '未说明', '不明确', '#n/a', 'n.a.', '缺失', '暂无', '空白'}
EXACT_GENERIC_NAMES = reader.GENERIC_NAMES | {
    '合计', '小计', '总计', '汇总', '关联方合计', '关联交易合计', '其他关联方合计',
    '其它关联方合计', '全部关联方', '所有关联方', '各关联方', '其他关联法人',
    '其它关联法人', '其他关联单位', '其它关联单位', '其他单位', '其它单位',
    '其他公司', '其它公司', '其他企业', '其它企业', '其他个人', '其它个人',
    '其他自然人', '其它自然人', '关联自然人', '关联法人', '自然人', '个人',
    '母公司及子公司', '子公司及其他关联方', '其他关联方及其子公司',
    '董事、监事及高级管理人员', '关键管理人员', '关键管理人员及其家属',
    '关键管理人员及其关系密切的家庭成员', '主要投资者个人及其家属'}
RESERVED_ID_WORDS = {'other', 'others', 'unknown', 'unk', 'missing', 'notavailable',
                     '其他', '其它', '合计', '关联方', '未知', '不详'}
GENERIC_PATTERNS = [
    ('aggregate_label', re.compile(r'^(?:所有|全部|各|上述|以下|本公司)?(?:关联方|关联企业|关联公司|交易方|其他关联方|其它关联方)?(?:合计|小计|总计|汇总|汇总数|总额)$')),
    ('other_party_group', re.compile(r'^(?:其他|其它|其余)(?:关联)?(?:方|法人|自然人|单位|公司|企业|个人|客户|供应商)(?:等|合计|及其子公司)?$')),
    ('management_family_group', re.compile(r'^(?:本公司|公司|上市公司)?(?:董事|监事|高级管理人员|关键管理人员|主要投资者个人)(?:、董事|、监事|、高级管理人员|及高级管理人员|及其家属|及其关系密切的家庭成员|及其控制的企业|及与其关系密切的家庭成员)+$')),
]


def name_category(value):
    if value in reader.SENTINELS or value in EXTRA_MISSING_NAMES:
        return 'missing_or_placeholder'
    if value in EXACT_GENERIC_NAMES:
        return 'generic_exact_label'
    for label, pattern in GENERIC_PATTERNS:
        if pattern.fullmatch(value):
            return label
    return 'unclassified_name_candidate'


def shape(value):
    """Only character classes and run lengths; never returns literal ID/name text."""
    runs = []
    for ch in value:
        k = 'D' if '0' <= ch <= '9' else 'L' if ch.isascii() and ch.isalpha() else 'W' if ch.isspace() else 'P' if ch.isascii() else 'U'
        if runs and runs[-1][0] == k:
            runs[-1][1] += 1
        else:
            runs.append([k, 1])
    return ''.join(k + '{' + str(n) + '}' for k, n in runs)[:300]


def id_profile(value):
    numeric_key = None
    try:
        number = Decimal(value)
        if not number.is_finite():
            conversion = 'nonfinite_numeric'
        elif number != number.to_integral_value():
            conversion = 'noninteger_numeric_Int64_incompatible'
        elif abs(number) >= Decimal(10) ** 15:
            conversion = 'integral_numeric_large_magnitude'
            numeric_key = format(number, 'f').split('.')[0]
        else:
            conversion = 'integral_numeric'
            numeric_key = format(number, 'f').split('.')[0]
    except InvalidOperation:
        conversion = 'nonnumeric_coercion_to_missing'
    if re.fullmatch(r'[0-9]+', value):
        category = 'ascii_digits_only'
    elif re.fullmatch(r'[A-Za-z0-9]+', value):
        category = 'ascii_alphanumeric'
    elif value.isascii():
        category = 'ascii_other'
    else:
        category = 'contains_nonascii'
    return category, conversion, numeric_key


def init_db(db):
    db.execute('PRAGMA journal_mode=OFF')
    db.execute('PRAGMA synchronous=OFF')
    db.execute('PRAGMA temp_store=MEMORY')
    db.execute('PRAGMA cache_size=-32768')
    db.executescript('''
        CREATE TABLE all_id_edges(year INTEGER,firm TEXT,party TEXT,nrows INTEGER,
                                 PRIMARY KEY(year,firm,party)) WITHOUT ROWID;
        CREATE TABLE all_name_edges(year INTEGER,firm TEXT,party TEXT,nrows INTEGER,
                                   PRIMARY KEY(year,firm,party)) WITHOUT ROWID;
        CREATE TABLE observations(year INTEGER,firm TEXT,party TEXT,name TEXT,pair TEXT,
                                  usable INTEGER,nrows INTEGER,
                                  PRIMARY KEY(year,firm,party,name)) WITHOUT ROWID;
        CREATE TABLE name_catalog(name TEXT PRIMARY KEY,category TEXT,display TEXT) WITHOUT ROWID;
        CREATE TABLE party_catalog(party TEXT PRIMARY KEY,category TEXT,conversion TEXT,
                                  numeric_key TEXT,display TEXT) WITHOUT ROWID;
    ''')


def read_volume(path, db, volume, secret):
    book = reader.Workbook(path, db, volume)
    counts = Counter()
    yearly = {}
    formats, lengths, patterns, conversions = Counter(), Counter(), Counter(), Counter()
    name_categories = Counter()
    batches = {k: [] for k in ['id', 'name', 'obs', 'names', 'parties']}

    @lru_cache(maxsize=131072)
    def digest(namespace, value):
        return hmac.new(secret, (namespace + '\0' + value).encode(), hashlib.sha256).hexdigest()

    def flush():
        db.executemany('INSERT INTO all_id_edges VALUES (?,?,?,?) ON CONFLICT(year,firm,party) DO UPDATE SET nrows=nrows+excluded.nrows', batches['id'])
        db.executemany('INSERT INTO all_name_edges VALUES (?,?,?,?) ON CONFLICT(year,firm,party) DO UPDATE SET nrows=nrows+excluded.nrows', batches['name'])
        db.executemany('INSERT INTO observations VALUES (?,?,?,?,?,?,?) ON CONFLICT(year,firm,party,name) DO UPDATE SET usable=MIN(usable,excluded.usable),nrows=nrows+excluded.nrows', batches['obs'])
        db.executemany('INSERT OR IGNORE INTO name_catalog VALUES (?,?,?)', batches['names'])
        db.executemany('INSERT OR IGNORE INTO party_catalog VALUES (?,?,?,?,?)', batches['parties'])
        for values in batches.values():
            values.clear()
        db.commit()

    header = None
    try:
        for physical, cells in book.rows():
            if physical == 1:
                header = {}
                for index, (value, _) in cells.items():
                    k = reader.clean(value).lstrip('\ufeff')
                    if k == 'RelatedPartyID':
                        k = 'RalatedPartyID'
                    if k in reader.FIELDS:
                        if k in header:
                            raise ValueError('DUPLICATE_REQUIRED_COLUMN')
                        header[k] = index
                if not {'Stkcd', 'Reptdt', 'Repart', 'RalatedPartyID'}.issubset(header):
                    raise ValueError('REQUIRED_RPT_COLUMNS_MISSING')
                continue
            if header is None:
                raise ValueError('FIRST_PHYSICAL_ROW_HEADER_MISSING')
            def cell(field):
                return cells.get(header[field], ('', 'n'))
            firm = reader.issuer(cell('Stkcd')[0])
            d, _ = reader.report_date(*cell('Reptdt'), book.date1904)
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
            if firm is None or d is None:
                counts['invalid_issuer_or_date_rows'] += 1
                continue
            counts['valid_issuer_date_rows'] += 1
            year = d.year
            yc = yearly.setdefault(year, Counter())
            yc['valid_issuer_date_rows'] += 1
            pid, unsafe = reader.party_key(*cell('RalatedPartyID'))
            norm = reader.norm_name(cell('Repart')[0])
            cat = name_category(norm)
            name_categories[cat] += 1
            yc['name_' + cat + '_rows'] += 1
            fh = digest('firm', firm)
            nh = digest('name', norm)
            # Preserve the Stage II exact-name diagnostic, including generic labels.
            if norm not in reader.SENTINELS:
                batches['name'].append((year, fh, nh, 1))
                batches['names'].append((nh, cat, norm))
            if pid.lower() in reader.SENTINELS:
                counts['missing_or_sentinel_id_rows'] += 1
                yc['missing_or_sentinel_id_rows'] += 1
            else:
                counts['candidate_id_rows'] += 1
                yc['candidate_id_rows'] += 1
                ph = digest('party', pid)
                fcat, conversion, numeric = id_profile(pid)
                formats[fcat] += 1
                lengths[str(len(pid))] += 1
                patterns[shape(pid)] += 1
                conversions[conversion] += 1
                generic_id = pid.casefold() in RESERVED_ID_WORDS
                for condition, key in [
                    (unsafe, 'unsafe_numeric_cell_id_rows'),
                    (generic_id, 'additional_reserved_id_word_rows'),
                    (bool(re.fullmatch(r'0[0-9]+', pid)), 'leading_zero_digit_id_rows'),
                    (pid.startswith(firm), 'id_starts_with_issuer_code_rows'),
                    (pid.endswith(firm), 'id_ends_with_issuer_code_rows'),
                    (firm in pid, 'id_contains_issuer_code_rows')]:
                    if condition:
                        counts[key] += 1
                        yc[key] += 1
                usable = cat == 'unclassified_name_candidate' and not unsafe and not generic_id
                if usable:
                    counts['candidate_id_with_clean_name_rows'] += 1
                    yc['candidate_id_with_clean_name_rows'] += 1
                batches['id'].append((year, fh, ph, 1))
                pair = digest('pair', json.dumps([pid, norm], ensure_ascii=False))
                batches['obs'].append((year, fh, ph, nh, pair, int(usable), 1))
                batches['names'].append((nh, cat, norm))
                batches['parties'].append((ph, fcat, conversion, '' if numeric is None else digest('numeric', numeric), pid))
            if len(batches['name']) + len(batches['id']) >= 10000:
                flush()
        flush()
    finally:
        book.close()
        digest.cache_clear()
    return {'volume_index': volume + 1, 'counts': dict(counts),
            'year_counts': {str(y): dict(c) for y, c in sorted(yearly.items())},
            'id_format_row_counts': dict(formats), 'id_length_row_counts': dict(lengths),
            'id_shape_row_counts_no_literal_characters': dict(patterns),
            'old_numeric_conversion_static_proxy_row_counts': dict(conversions),
            'name_category_row_counts': dict(name_categories)}


def construct_views(db):
    db.executescript('''
        CREATE INDEX observations_party_year ON observations(year,party,name,firm);
        CREATE VIEW clean_id_edges AS
          SELECT DISTINCT year,firm,party FROM observations WHERE usable=1;
        CREATE VIEW compound_edges AS
          SELECT DISTINCT year,firm,pair AS party FROM observations WHERE usable=1;
        CREATE VIEW clean_name_edges AS
          SELECT e.year,e.firm,e.party FROM all_name_edges e JOIN name_catalog n ON e.party=n.name
          WHERE n.category='unclassified_name_candidate';
        CREATE TABLE conflicting_id_year AS
          SELECT year,party FROM observations WHERE usable=1
          GROUP BY year,party HAVING COUNT(DISTINCT name)>1;
        CREATE UNIQUE INDEX conflicting_id_year_index ON conflicting_id_year(year,party);
        CREATE VIEW unambiguous_id_edges AS
          SELECT e.year,e.firm,e.party FROM clean_id_edges e
          WHERE NOT EXISTS(SELECT 1 FROM conflicting_id_year c WHERE c.year=e.year AND c.party=e.party);
    ''')


def scalar(db, sql, params=()):
    return int(db.execute(sql, params).fetchone()[0] or 0)


def identity_scope(db, lower, upper):
    counts = {
        'first_report_year': lower, 'last_report_year': upper,
        'candidate_ids': scalar(db, 'SELECT COUNT(DISTINCT party) FROM observations WHERE year BETWEEN ? AND ?', (lower, upper)),
        'ids_multiple_names_after_name_exclusions': scalar(db, '''SELECT COUNT(*) FROM
          (SELECT party FROM observations WHERE usable=1 AND year BETWEEN ? AND ?
           GROUP BY party HAVING COUNT(DISTINCT name)>1)''', (lower, upper)),
        'ids_multiple_issuers_after_name_exclusions': scalar(db, '''SELECT COUNT(*) FROM
          (SELECT party FROM observations WHERE usable=1 AND year BETWEEN ? AND ?
           GROUP BY party HAVING COUNT(DISTINCT firm)>1)''', (lower, upper)),
        'ids_multiple_names_in_any_one_year_after_name_exclusions': scalar(db, '''SELECT COUNT(DISTINCT party)
          FROM conflicting_id_year WHERE year BETWEEN ? AND ?''', (lower, upper)),
    }
    rows = []
    for (year,) in db.execute('SELECT DISTINCT year FROM observations WHERE year BETWEEN ? AND ? ORDER BY year', (lower, upper)):
        p = (year,)
        row = {'year': year,
            'candidate_ids': scalar(db, 'SELECT COUNT(DISTINCT party) FROM observations WHERE year=?', p),
            'clean_name_ids': scalar(db, 'SELECT COUNT(DISTINCT party) FROM observations WHERE year=? AND usable=1', p),
            'clean_name_shared_ids': scalar(db, '''SELECT COUNT(*) FROM
              (SELECT party FROM observations WHERE year=? AND usable=1 GROUP BY party HAVING COUNT(DISTINCT firm)>1)''', p),
            'ids_multiple_clean_names_same_year': scalar(db, 'SELECT COUNT(*) FROM conflicting_id_year WHERE year=?', p),
            'issuer_id_pairs_multiple_clean_names_same_year': scalar(db, '''SELECT COUNT(*) FROM
              (SELECT firm,party FROM observations WHERE year=? AND usable=1 GROUP BY firm,party HAVING COUNT(DISTINCT name)>1)''', p),
            'shared_ids_multiple_clean_names_same_year': scalar(db, '''SELECT COUNT(*) FROM
              (SELECT o.party FROM observations o JOIN conflicting_id_year c ON c.year=o.year AND c.party=o.party
               WHERE o.year=? AND o.usable=1 GROUP BY o.party HAVING COUNT(DISTINCT o.firm)>1)''', p),
            'clean_incidence_edges_affected_by_name_disagreements': scalar(db, '''SELECT COUNT(*) FROM
              clean_id_edges e JOIN conflicting_id_year c ON c.year=e.year AND c.party=e.party WHERE e.year=?''', p),
            'issuers_affected_by_name_disagreements': scalar(db, '''SELECT COUNT(DISTINCT e.firm) FROM
              clean_id_edges e JOIN conflicting_id_year c ON c.year=e.year AND c.party=e.party WHERE e.year=?''', p),
            'clean_source_rows_affected_by_name_disagreements': scalar(db, '''SELECT SUM(o.nrows) FROM
              observations o JOIN conflicting_id_year c ON c.year=o.year AND c.party=o.party WHERE o.year=? AND o.usable=1''', p),
            'clean_names_with_multiple_ids_same_year': scalar(db, '''SELECT COUNT(*) FROM
              (SELECT name FROM observations WHERE year=? AND usable=1 GROUP BY name HAVING COUNT(DISTINCT party)>1)''', p)}
        rows.append(row)
    return {'scope_counts': counts, 'annual_counts': rows}


def name_hubs(db, limit=20):
    aggregate, local = [], []
    for year in [2010, 2018, 2021, 2022]:
        names = db.execute('''SELECT e.party,COUNT(DISTINCT e.firm),n.category,n.display
          FROM all_name_edges e JOIN name_catalog n ON n.name=e.party WHERE e.year=?
          GROUP BY e.party ORDER BY COUNT(DISTINCT e.firm) DESC,e.party LIMIT ?''', (year, limit)).fetchall()
        aggregate.append({'year': year, 'top_names_no_names_or_ids': [
            {'rank': i + 1, 'distinct_issuer_degree': degree, 'category': category,
             'name_character_count': len(display), 'name_character_shape': shape(display)}
            for i, (_, degree, category, display) in enumerate(names)]})
        local.append({'year': year, 'names': [
            {'rank': i + 1, 'distinct_issuer_degree': degree, 'category': category,
             'normalized_source_name_PRIVATE': display}
            for i, (_, degree, category, display) in enumerate(names)]})
    return aggregate, local


def local_identity_examples(db, limit=20):
    groups = db.execute('''SELECT c.year,c.party,COUNT(DISTINCT o.firm),COUNT(DISTINCT o.name)
      FROM conflicting_id_year c JOIN observations o ON o.year=c.year AND o.party=c.party
      WHERE o.usable=1 AND c.year BETWEEN 2010 AND 2022
      GROUP BY c.year,c.party ORDER BY COUNT(DISTINCT o.firm) DESC,c.year DESC,c.party LIMIT ?''', (limit,)).fetchall()
    result = []
    for year, party, firms, names in groups:
        pid = db.execute('SELECT display FROM party_catalog WHERE party=?', (party,)).fetchone()[0]
        displays = db.execute('''SELECT DISTINCT n.display FROM observations o
          JOIN name_catalog n ON n.name=o.name WHERE o.year=? AND o.party=? AND o.usable=1
          ORDER BY n.display LIMIT 10''', (year, party)).fetchall()
        result.append({'report_year': year, 'provider_id_PRIVATE': pid,
                       'distinct_issuer_count': firms, 'distinct_clean_name_count': names,
                       'normalized_source_names_PRIVATE': [r[0] for r in displays]})
    return result


def numeric_proxy_stats(db):
    return {
        'unique_id_format_counts': {k: n for k, n in db.execute('SELECT category,COUNT(*) FROM party_catalog GROUP BY category')},
        'unique_id_conversion_static_proxy_counts': {k: n for k, n in db.execute('SELECT conversion,COUNT(*) FROM party_catalog GROUP BY conversion')},
        'exact_numeric_value_groups_with_multiple_lexical_ids': scalar(db, '''SELECT COUNT(*) FROM
          (SELECT numeric_key FROM party_catalog WHERE numeric_key<>'' GROUP BY numeric_key HAVING COUNT(*)>1)'''),
        'note': 'Exact Decimal parsing is a static diagnostic proxy. It does not reproduce pandas float conversion or prove execution of the old builder.'}


def old_builder_evidence(path):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    return {
        'to_numeric_errors_coerce_call_present': any(isinstance(n.func, ast.Attribute) and n.func.attr == 'to_numeric' and any(k.arg == 'errors' and isinstance(k.value, ast.Constant) and k.value.value == 'coerce' for k in n.keywords) for n in calls),
        'astype_Int64_call_present': any(isinstance(n.func, ast.Attribute) and n.func.attr == 'astype' and any(isinstance(a, ast.Constant) and a.value == 'Int64' for a in n.args) for n in calls),
        'old_builder_executed': False}


def csv_write(path, rows, fields=None):
    import csv
    fields = fields or list(rows[0])
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


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
    output = reader.checked_output(args.output_dir, boundary)
    reference = json.loads(Path(args.reference).read_text(encoding='utf-8'))
    if not args.local_test and Path(args.reference).resolve() != KIT / 'PRIOR_AUDIT_REFERENCE.json':
        raise ValueError('PRODUCTION_REFERENCE_MUST_BE_FROZEN_PACKAGE_FILE')
    identities = reference['expected_sources']
    checked = []
    for original in identities:
        relative = Path(original['path']).relative_to(PROJECT)
        path = source_root / relative
        if not path.is_file() or not path.resolve().is_relative_to(source_root):
            raise ValueError('EXPECTED_SOURCE_MISSING_OR_OUTSIDE_ROOT')
        print('Checking frozen source: ' + path.name, flush=True)
        if reader.sha(path) != original['sha256'] or path.stat().st_size != original['size_bytes']:
            raise ValueError('SOURCE_IDENTITY_DIFFERS_FROM_STAGE_II')
        checked.append({'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': original['sha256']})
    output.mkdir(parents=True)
    start = time.time()
    per_volume = []
    with tempfile.TemporaryDirectory(prefix='private_focus_', dir=output) as private:
        db = sqlite3.connect(str(Path(private) / 'private_cache.sqlite'))
        try:
            init_db(db)
            secret = secrets.token_bytes(32)
            for volume, filename in enumerate(reader.VOLUMES):
                print('Reading volume ' + str(volume + 1) + ' of 3', flush=True)
                per_volume.append(read_volume(source_root / 'data/raw/csmar' / filename, db, volume, secret))
            construct_views(db)
            profiles = {name: reader.annual_structure(db, table) for name, table in [
                ('stage_ii_id_candidate_reproduction', 'all_id_edges'),
                ('stage_ii_name_diagnostic_reproduction', 'all_name_edges'),
                ('id_with_clean_name_candidate', 'clean_id_edges'),
                ('id_plus_exact_clean_name_candidate', 'compound_edges'),
                ('id_excluding_same_year_name_disagreement_candidate', 'unambiguous_id_edges'),
                ('clean_exact_name_diagnostic_ONLY', 'clean_name_edges')]}
            for name, refkey in [('stage_ii_id_candidate_reproduction', 'expected_annual_id_structure'),
                                 ('stage_ii_name_diagnostic_reproduction', 'expected_annual_name_structure')]:
                if profiles[name] != reference[refkey]:
                    raise ValueError('STAGE_II_ANNUAL_REPRODUCTION_MISMATCH')
            primary = identity_scope(db, 2010, 2022)
            all_years = identity_scope(db, 2001, 2024)
            hubs, local_hubs = name_hubs(db)
            numeric = numeric_proxy_stats(db)
            if args.write_local_review:
                reader.write_json(output / 'LOCAL_PRIVATE_IDENTITY_REVIEW.json', {
                    'PRIVATE_LOCAL_ONLY_DO_NOT_UPLOAD': True,
                    'purpose': 'Bounded local inspection of hub names and same-year ID/name disagreements. No labels or scores.',
                    'top_name_hubs': local_hubs,
                    'same_year_id_name_disagreements': local_identity_examples(db)})
        finally:
            db.close()
    total = Counter()
    for volume in per_volume:
        total.update(volume['counts'])
    for key in ['data_rows', 'valid_issuer_date_rows', 'candidate_id_rows', 'missing_or_sentinel_id_rows']:
        if total.get(key, 0) != reference['expected_total_counts'].get(key, 0):
            raise ValueError('STAGE_II_ROW_COUNT_REPRODUCTION_MISMATCH')
    after = {row['path']: reader.sha(row['path']) == row['sha256'] for row in checked}
    if not all(after.values()):
        raise ValueError('SOURCE_CHANGED_DURING_FOCUSED_AUDIT')
    summary = {
        'status': 'IDENTITY_FOCUS_COMPLETE_REVIEW_REQUIRED', 'version': VERSION,
        'training_started': False, 'labels_or_scores_read': False,
        'graph_exported': False, 'global_party_id_semantics_verified': False,
        'prior_return_sha256': reference['return_sha256'],
        'prior_annual_and_row_count_reproduction': True,
        'source_recheck_match': after, 'total_counts': dict(total), 'per_volume': per_volume,
        'primary_2010_2022_identity_diagnostics': primary,
        'all_2001_2024_identity_diagnostics': all_years,
        'identity_static_conversion_diagnostics': numeric,
        'old_builder_static_evidence': old_builder_evidence(source_root / 'scripts/p1_v1_integration/scripts/build_kg_e7_trade.py'),
        'name_hubs_no_names_or_ids': hubs,
        'annual_profiles_NOT_selected_training_graphs': profiles,
        'private_local_review_written': args.write_local_review,
        'private_local_review_included_in_return': False,
        'temporary_database_retained': False,
        'raw_names_ids_rows_in_return': False,
        'interpretation_limits': [
            'All-year name disagreement does not distinguish renaming, aliases and collisions.',
            'Even same-year agreement is empirical consistency, not a globally validated legal-entity identifier.',
            'Clean means the explicitly frozen placeholder/group rules did not match; it is not a legal-entity assertion.',
            'ID+name and collision-exclusion profiles are diagnostics; neither is silently selected for training.',
            'The all-2001-2024 profile cannot set a 2010-2022 training key or exclusion rule.',
            'Repeated issuer-party-year incidences are not necessarily duplicate transactions.',
            'Raw-source company coverage is not frozen-baseline eligible-company coverage.',
            'Reptdt is a statistical cutoff date; publication/availability times remain unestablished.'],
        'elapsed_seconds': round(time.time() - start, 3)}
    reader.write_json(output / 'FOCUS_SUMMARY.json', summary)
    reader.write_json(output / 'SOURCE_IDENTITIES.json', {'sources': checked, 'source_recheck_match': after})
    csv_write(output / 'ANNUAL_IDENTITY.csv', all_years['annual_counts'])
    flat = [{'key_basis': basis, **row} for basis, rows in profiles.items() for row in rows]
    fields = [key for key in flat[0] if key != 'top_20_party_degrees_no_identifiers']
    csv_write(output / 'ANNUAL_PROFILES.csv', flat, fields)
    reader.write_json(output / 'RUNNER_IDENTITIES.json', {
        p.name: {'size_bytes': p.stat().st_size, 'sha256': reader.sha(p)}
        for p in [KIT / name for name in RUNNER_FILES] if p.is_file() and not p.is_symlink()})
    (output / 'README_RETURN.txt').write_text(
        'Private RPT identity diagnostics. No training, graph export, labels or scores.\n'
        'Only aggregate diagnostics and source/runner identities are returned.\n'
        'LOCAL_PRIVATE_IDENTITY_REVIEW.json, if created, remains on Tencent and is excluded.\n'
        'IDENTITY_FOCUS_COMPLETE_REVIEW_REQUIRED is not permission to start training.\n', encoding='utf-8')
    target = output / 'E7_IDENTITY_RETURN.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in RETURN_FILES:
            z.write(output / name, arcname=name)
    print('IDENTITY_FOCUS_COMPLETE_REVIEW_REQUIRED', flush=True)
    print('Return ZIP: ' + str(target), flush=True)
    if args.write_local_review:
        print('Local private review (excluded from ZIP): ' + str(output / 'LOCAL_PRIVATE_IDENTITY_REVIEW.json'), flush=True)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', default=str(PROJECT))
    p.add_argument('--output-dir', required=True)
    p.add_argument('--reference', default=str(KIT / 'PRIOR_AUDIT_REFERENCE.json'))
    p.add_argument('--write-local-review', action='store_true')
    p.add_argument('--local-test', action='store_true')
    return run(p.parse_args())


if __name__ == '__main__':
    try:
        if sys.version_info < (3, 10):
            raise ValueError('PYTHON_3_10_OR_NEWER_REQUIRED')
        main()
    except Exception as error:
        message = str(error)
        status = message if isinstance(error, ValueError) and re.fullmatch(r'[A-Z0-9_]+', message) else 'FOCUSED_AUDIT_ERROR'
        print(json.dumps({'status': status, 'error_type': type(error).__name__}), file=sys.stderr)
        raise SystemExit(1)
