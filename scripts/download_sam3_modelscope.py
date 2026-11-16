"""Pinned public ModelScope SAM3 file, range-resumable, verified against repository SHA256."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import requests,json,hashlib,time
ROOT=Path("/home/tione/notebook/home/keithanwang/vlm-vla/gafu")
BASE=ROOT/"sceneblocker/baseline"
REV="f8ee5d16d5c30d0e87b1d22cbd358ce97761df56"
SIZE=3450062241
SHA="9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
URL=f"https://modelscope.cn/models/facebook/sam3/resolve/{REV}/sam3.pt"
dest=BASE/"weights/facebook/sam3";dest.mkdir(parents=True,exist_ok=True)
target=dest/"sam3.pt"
CHUNK=32*1024**2
def fetch(start):
    end=min(start+CHUNK,SIZE)-1
    part=dest/f"sam3.pt.part-{start}"
    if part.exists() and part.stat().st_size==end-start+1:return
    for attempt in range(4):
        try:
            with requests.get(URL+f"?range_start={start}",headers={"Range":f"bytes={start}-{end}"},stream=True,timeout=(10,90)) as r:
                r.raise_for_status()
                if r.status_code!=206 or r.headers.get("Content-Range")!=f"bytes {start}-{end}/{SIZE}":
                    raise ValueError("Server did not honor byte range")
                tmp=part.with_suffix(part.suffix+".tmp")
                written=0
                with tmp.open("wb") as f:
                    for b in r.iter_content(1024**2):
                        written+=len(b)
                        if written>end-start+1:raise ValueError("Oversized range")
                        f.write(b)
                if written!=end-start+1:raise ValueError("Incomplete range")
                tmp.replace(part);return
        except (requests.RequestException,ValueError):
            if attempt==3:raise
            time.sleep(attempt+1)
if not target.exists():
    jobs=list(range(0,SIZE,CHUNK))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(fetch,a) for a in jobs]
        for i,f in enumerate(as_completed(futures),1):
            f.result()
            if i%8==0:print("ranges",i,"/",len(jobs),flush=True)
    temp=target.with_suffix(".assembling")
    with temp.open("wb") as f:
        for start in jobs:
            with (dest/f"sam3.pt.part-{start}").open("rb") as inp:
                for b in iter(lambda:inp.read(8*1024**2),b""):f.write(b)
    temp.replace(target)
h=hashlib.sha256()
with target.open("rb") as f:
    for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
if target.stat().st_size!=SIZE or h.hexdigest()!=SHA:raise ValueError("Checkpoint SHA256 mismatch")
report={"repo":"facebook/sam3","provider":"ModelScope","revision":REV,"file":str(target),"size":SIZE,"sha256":h.hexdigest(),"status":"sha256_verified_against_modelscope_metadata","cross_provider_hash_verified":False}
(BASE/"reports/sam3-weight.json").write_text(json.dumps(report,indent=2))
for start in range(0,SIZE,CHUNK):(dest/f"sam3.pt.part-{start}").unlink(missing_ok=True)
print(json.dumps(report),flush=True)
