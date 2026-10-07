#!/usr/bin/env python3
"""Portable, read-only E7 aggregate check or source-identity inventory."""
from pathlib import Path
import argparse, json, sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-sources',action='store_true')
    args=parser.parse_args()
    if args.list_sources:
        records=json.loads((ROOT/'provenance/source_hashes.json').read_text())['records']
        result=[r for r in records if r['public_path'].startswith('code/e7/')]
        print(json.dumps(result,indent=2));return
    sys.path.insert(0,str(ROOT/'tools'))
    from verify_e7 import verify
    print(json.dumps(verify(ROOT),indent=2))

if __name__=='__main__':main()
