"""Workspace-limited immutable artifacts and reproducible source binding."""
import hashlib,json,os
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(os.environ.get("SCENEGEN_BASE", Path(__file__).resolve().parents[1])).resolve()
ROOT=Path(os.environ.get("SCENEGEN_ROOT", BASE)).resolve()
if not BASE.is_relative_to(ROOT): raise ValueError("SCENEGEN_BASE must be inside SCENEGEN_ROOT")

def inside(path):
    p=Path(path).expanduser().resolve()
    if not p.is_relative_to(ROOT): raise ValueError("Path outside authorized root")
    return p

def sha256(path):
    h=hashlib.sha256()
    with inside(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024**2),b""): h.update(block)
    return h.hexdigest()

def reference(path):
    p=inside(path)
    return {"path":str(p),"sha256":sha256(p)}

def verify(ref):
    if reference(ref["path"])!=ref: raise ValueError("Changed input artifact")
    return inside(ref["path"])

def read_json(path):
    return json.loads(inside(path).read_text())

def write_json(path,obj):
    p=inside(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x") as f: json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)

def new_run(path):
    p=inside(path);p.mkdir(parents=True,exist_ok=False);return p

def utc():
    return datetime.now(timezone.utc).isoformat()
