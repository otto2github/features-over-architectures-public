#!/usr/bin/env python3
"""Tencent-only, outcome-free conservative RPT record-incidence construction."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from functools import lru_cache
import gzip
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

sys.dont_write_bytecode = True
import e7_xlsx_reader as reader
from e7_identity_focus import name_category, RESERVED_ID_WORDS
from e7_prep_common import KIT, YEARS, new_output, require, sha, verify_records, write_json, zip_exact

PROJECT = Path('/path/to/private_source_project')


@lru_cache(maxsize=131072)
def token(namespace, value):
    # Deterministic private record keys; this is pseudonymization, not anonymization.
    body = json.dumps([namespace, value], ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(body.encode('utf-8')).hexdigest()


def init_db(db):
    db.executescript('''
      PRAGMA journal_mode=OFF;
      PRAGMA synchronous=OFF;
      PRAGMA temp_store=MEMORY;
      CREATE TABLE obs(year INTEGER,firm TEXT,id TEXT,name TEXT,pair TEXT,nrows INTEGER,
        PRIMARY KEY(year,firm,id,name)) WITHOUT ROWID;
    ''')


def read_volume(path, db, volume):
    book = reader.Workbook(path, db, volume)
    counts, batch = Counter(), []
    def flush():
        db.executemany('''INSERT INTO obs VALUES (?,?,?,?,?,?)
          ON CONFLICT(year,firm,id,name) DO UPDATE SET nrows=nrows+excluded.nrows''', batch)
        batch.clear()
        db.commit()
    header = None
    try:
        for physical, cells in book.rows():
            if physical == 1:
                header = {}
                for index, (value, _) in cells.items():
                    key = reader.clean(value).lstrip('\ufeff')
                    if key == 'RelatedPartyID':
                        key = 'RalatedPartyID'
                    if key in reader.FIELDS:
                        require(key not in header, 'DUPLICATE_RPT_COLUMN')
                        header[key] = index
                require({'Stkcd','Reptdt','Repart','RalatedPartyID'} <= set(header),
                        'REQUIRED_RPT_COLUMNS_MISSING')
                continue
            require(header is not None, 'FIRST_PHYSICAL_ROW_HEADER_MISSING')
            def cell(key):
                return cells.get(header[key], ('', 'n'))
            firm = reader.issuer(cell('Stkcd')[0])
            date, _ = reader.report_date(*cell('Reptdt'), book.date1904)
            if physical in {2, 3}:
                require(firm is None or date is None, 'METADATA_ROW_LOOKS_LIKE_RECORD')
                counts['metadata_rows_skipped'] += 1
                continue
            if not any(value for value, _ in cells.values()):
                counts['empty_rows'] += 1
                continue
            counts['data_rows'] += 1
            if counts['data_rows'] % 100000 == 0:
                print(f'Volume {volume+1}: {counts["data_rows"]} rows read', flush=True)
            if firm is None or date is None:
                counts['invalid_issuer_or_date_rows'] += 1
                continue
            counts['valid_issuer_date_rows'] += 1
            pid, unsafe = reader.party_key(*cell('RalatedPartyID'))
            name = reader.norm_name(cell('Repart')[0])
            if pid.lower() in reader.SENTINELS:
                counts['missing_or_sentinel_id_rows'] += 1
                continue
            counts['candidate_id_rows'] += 1
            usable = (not unsafe and pid.casefold() not in RESERVED_ID_WORDS
                      and name_category(name) == 'unclassified_name_candidate')
            if usable:
                counts['candidate_id_with_clean_name_rows'] += 1
            if date.year not in YEARS:
                counts['outside_primary_window_candidate_rows'] += 1
                continue
            counts['primary_candidate_id_rows'] += 1
            if not usable:
                counts['primary_excluded_name_or_id_rows'] += 1
                continue
            batch.append((date.year, firm, token('provider_id', pid), token('name', name),
                          token('e7_rpt_record_v1', (pid, name)), 1))
            if len(batch) >= 10000:
                flush()
        flush()
    finally:
        book.close()
        token.cache_clear()
    return dict(counts)


def select_edges(db):
    db.executescript('''
      CREATE INDEX obs_year_id ON obs(year,id,name);
      CREATE TABLE conflicts AS SELECT year,id FROM obs
        GROUP BY year,id HAVING COUNT(DISTINCT name)>1;
      CREATE UNIQUE INDEX conflict_index ON conflicts(year,id);
      CREATE VIEW selected AS SELECT DISTINCT o.year,o.firm,o.pair AS party FROM obs o
        WHERE NOT EXISTS(SELECT 1 FROM conflicts c WHERE c.year=o.year AND c.id=o.id);
    ''')
    return reader.annual_structure(db, 'selected')


def write_incidence(path, db):
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as text:
                writer = csv.writer(text, lineterminator='\n')
                writer.writerow(['year', 'firm_code', 'party_key'])
                for row in db.execute('SELECT year,firm,party FROM selected ORDER BY year,firm,party'):
                    writer.writerow(row)


def run(output):
    started = time.monotonic()
    reference = json.loads((KIT / 'INPUT_BINDINGS.json').read_text())
    records = reference['tencent_sources']
    verify_records(records)
    counts, per_volume = Counter(), []
    with tempfile.TemporaryDirectory(prefix='private_work_', dir=output) as work:
        with sqlite3.connect(Path(work) / 'working.sqlite') as db:
            init_db(db)
            for volume, filename in enumerate(reader.VOLUMES):
                values = read_volume(PROJECT / 'data/raw/csmar' / filename, db, volume)
                counts.update(values)
                per_volume.append({'volume_index': volume+1, 'counts': values})
            for key, expected in reference['focus_total_count_checks'].items():
                require(counts[key] == expected, 'IDENTITY_COUNT_REPRODUCTION_FAILED:' + key)
            require(counts['primary_candidate_id_rows'] == reference['primary_candidate_id_rows'],
                    'PRIMARY_CANDIDATE_ROWS_MISMATCH')
            profiles = select_edges(db)
            require(profiles == reference['selected_annual_profile_oracle'],
                    'CONSERVATIVE_GRAPH_PROFILE_REPRODUCTION_FAILED')
            selected_count = sum(p['unique_issuer_party_year_edges'] for p in profiles)
            require(selected_count == reference['selected_primary_edge_total'],
                    'PRIMARY_SELECTED_EDGE_TOTAL_MISMATCH')
            conflict_counts = [{'year': y, 'conflicting_id_count': n} for y,n in
                               db.execute('SELECT year,COUNT(*) FROM conflicts GROUP BY year ORDER BY year')]
            incidence = output / 'e7_record_incidence.csv.gz'
            write_incidence(incidence, db)
    verify_records(records)
    summary = {
      'status': 'PRIVATE_RECORD_GRAPH_BUILT_ALIGNMENT_PENDING', 'training_started': False,
      'labels_or_model_scores_read': False, 'global_legal_party_id_semantics_verified': False,
      'graph_kind': 'conservative RPT-derived ID/name record-incidence; not legal-entity crosswalk',
      'graph_rule_id': reference['graph_rule_id'],
      'protocol_sha256': sha(KIT / 'E7_GRAPH_PREP_PROTOCOL_v1.md'),
      'identity_return_sha256': reference['identity_return_sha256'],
      'source_identity_rechecked': True, 'source_identities': records,
      'total_counts': dict(counts), 'per_volume': per_volume,
      'annual_profiles_before_frozen_feature_panel_alignment': profiles,
      'same_year_conflicting_id_counts': conflict_counts,
      'incidence_file': {'name': incidence.name, 'sha256': sha(incidence),
                         'size_bytes': incidence.stat().st_size, 'rows': selected_count},
      'privacy': 'Private licensed derivative; no raw names or provider IDs; never a public-release artifact',
      'timing': 'Reptdt report year; historical disclosure availability not verified',
      'temporary_database_retained': False, 'elapsed_seconds': time.monotonic()-started}
    write_json(output / 'PRIVATE_BRIDGE_MANIFEST.json', summary)
    zip_exact(output / 'E7_PRIVATE_BRIDGE.zip', output,
              ['PRIVATE_BRIDGE_MANIFEST.json','e7_record_incidence.csv.gz'])
    print('PRIVATE_RECORD_GRAPH_DONE; ALIGNMENT_PENDING; NO_MODEL_FITTED', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True, type=Path)
    args = ap.parse_args()
    output = new_output(args.output_dir, PROJECT)
    run(output)


if __name__ == '__main__':
    main()
