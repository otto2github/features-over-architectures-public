#!/usr/bin/env python3
"""Verify the actual public package and retained aggregate scientific record."""
from pathlib import Path
import hashlib,json,sys,io,contextlib
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def require(ok,message):
 if not ok:raise AssertionError(message)
def main():
 manifest=json.loads((ROOT/'FILE_SHA256.json').read_text())
 actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name!='FILE_SHA256.json' and '__pycache__' not in p.parts and '.venv' not in p.parts}
 require(actual==set(manifest['files']),'MANIFEST_MEMBER_SET')
 for name,item in manifest['files'].items():
  p=ROOT/name;require(p.stat().st_size==item['size_bytes'] and sha(p)==item['sha256'],'MANIFEST_FILE:'+name)
 release=json.loads((ROOT/'provenance/release.json').read_text())
 require(release['edition']=='R1' and release['science_changed'] is True,'CURRENT_RELEASE_SCOPE')
 require((release['new_fits'],release['new_predictions'],release['new_empirical_resampling_draws'])==(40,40,4000),'SEPARATE_E7_AND_W2_EXTENSION_SCOPE')
 ledger=json.loads((ROOT/'provenance/source_hashes.json').read_text())
 for item in ledger['records']:
  require(sha(ROOT/item['public_path'])==item['public_sha256'],'PUBLIC_SOURCE_HASH:'+item['public_path'])
  if item['representation']=='byte_identical':require(item['original_sha256']==item['public_sha256'],'IDENTITY_ROLE')
 paths=json.loads((ROOT/'documentation/archive_paths.json').read_text())
 require(all((ROOT/p).exists() for p in paths.values()),'PUBLIC_NAVIGATION_TARGET')
 require(not list(ROOT.rglob('*.zip')),'NO_NESTED_ARCHIVE_DEPENDENCY')
 from verify_scientific_record import main as scientific
 out=io.StringIO()
 with contextlib.redirect_stdout(out):scientific()
 result=json.loads(out.getvalue())
 from verify_e7 import verify as e7_verify
 e7=e7_verify(ROOT)
 from verify_w2 import verify as w2_verify
 w2=w2_verify(ROOT)
 print(json.dumps({'status':'R1_PUBLIC_REPRODUCIBILITY_PASS','e7_checks':e7,'w2_checks':w2,'manifest_files':len(actual),'source_copy_records':len(ledger['records']),'scientific_checks':result,'release_identifiers_claimed':release['version_doi'] is not None,'verification_scope':'Distributed bytes, existing aggregate arithmetic and table grids; not private empirical reconstruction.'},indent=2))
if __name__=='__main__':main()
