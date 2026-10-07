"""Shared file-boundary and identity checks for the private W2 kit (adapted from the executed E7 kit)."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True
KIT = Path(__file__).resolve().parent
YEARS = list(range(2010, 2023))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in [path,*path.parents]),'JSON_WRITE_SYMLINK_REJECTED')
    if os.environ.get('W2_WRITE_BOUNDARY'):
        safe_path(path,Path(os.environ['W2_WRITE_BOUNDARY']))
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('x',encoding='utf-8') as stream:
        json.dump(obj,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)


def safe_path(path, boundary):
    path, boundary = Path(path).absolute(), Path(boundary).absolute()
    require(not any(p.is_symlink() for p in [path, *path.parents]),
            'SYMLINK_PATH_REJECTED:' + str(path))
    require(boundary.resolve() == boundary and path.resolve().is_relative_to(boundary)
            and path != boundary, 'WRITE_BOUNDARY_REJECTED:' + str(path))
    return path


def new_output(path, boundary):
    path = safe_path(path, boundary)
    require(not path.exists(), 'OUTPUT_ALREADY_EXISTS:' + str(path))
    path.mkdir(parents=True, mode=0o700)
    return path


def verify_package():
    manifest = json.loads((KIT / 'PACKAGE_SHA256.json').read_text())
    for name, expected in manifest['files'].items():
        p = KIT / name
        require(p.is_file() and not p.is_symlink() and p.resolve().parent == KIT,
                'PACKAGE_FILE_REJECTED:' + name)
        require(sha(p) == expected, 'PACKAGE_HASH_MISMATCH:' + name)
    return manifest


def verify_records(records):
    for rec in records:
        p = Path(rec['path'])
        require(p.is_file(), 'BOUND_SOURCE_MISSING:' + str(p))
        require(p.stat().st_size == rec['size_bytes'] and sha(p) == rec['sha256'],
                'BOUND_SOURCE_IDENTITY_MISMATCH:' + str(p))


def zip_exact(path, directory, names):
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as book:
        for name in names:
            p = Path(directory) / name
            require(p.is_file() and not p.is_symlink(), 'ZIP_INPUT_REJECTED')
            book.write(p, name)
    with zipfile.ZipFile(path) as book:
        require(book.testzip() is None and book.namelist() == names, 'RETURN_ZIP_INVALID')


def publish(source, target, boundary):
    target = safe_path(target, boundary)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='publish_', delete=False) as stream:
        temporary = Path(stream.name)
        with Path(source).open('rb') as incoming:
            shutil.copyfileobj(incoming, stream)
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
