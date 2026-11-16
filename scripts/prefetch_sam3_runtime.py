"""Parallel fetch of official PyTorch/PyPI wheels with upstream SHA256 verification."""
import requests,json,re,hashlib,time
from pathlib import Path
from urllib.parse import urljoin,urlsplit,parse_qs,unquote
from concurrent.futures import ThreadPoolExecutor,as_completed
BASE=Path("/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/baseline")
DEST=BASE/"downloads/sam3-runtime";DEST.mkdir(parents=True,exist_ok=True)
items=[]
packages={"nvidia-cufft-cu12":"11.3.0.4","nvidia-curand-cu12":"10.3.7.77","nvidia-cusolver-cu12":"11.7.1.2",
 "nvidia-cusparse-cu12":"12.5.4.2","nvidia-cusparselt-cu12":"0.6.3","nvidia-nccl-cu12":"2.26.2",
 "nvidia-nvtx-cu12":"12.6.77","nvidia-nvjitlink-cu12":"12.6.85","nvidia-cufile-cu12":"1.11.1.6","triton":"3.3.1"}
def metadata(pair):
 name,version=pair
 d=requests.get(f"https://pypi.org/pypi/{name}/{version}/json",timeout=30).json()
 files=[v for v in d["urls"] if v["filename"].endswith(".whl") and "manylinux" in v["filename"] and "x86_64" in v["filename"] and ("py3-none" in v["filename"] or "cp312-cp312" in v["filename"])]
 if not files:raise ValueError("No runtime wheel "+name)
 v=files[0];return {"filename":v["filename"],"url":v["url"],"size":v["size"],"sha256":v["digests"]["sha256"],"source":"PyPI"}
with ThreadPoolExecutor(max_workers=8) as p:items=list(p.map(metadata,packages.items()))
for package,version in [("torch","2.7.1"),("torchvision","0.22.1")]:
 page=f"https://download.pytorch.org/whl/cu126/{package}/"
 html=requests.get(page,timeout=30).text
 links=re.findall(r'href="([^"]+)"',html)
 links=[urljoin(page,x.replace("&amp;","&")) for x in links if f"{package}-{version}%2Bcu126-cp312-cp312" in x and "manylinux_2_28_x86_64" in x]
 if not links:raise ValueError("Missing PyTorch wheel "+package)
 url=links[0];sha=parse_qs(urlsplit(url).fragment)["sha256"][0]
 response=requests.get(url.split("#")[0],headers={"Range":"bytes=0-0"},stream=True,timeout=30)
 cr=response.headers.get("Content-Range","");response.close()
 total=int(cr.split("/")[-1])
 items.append({"filename":unquote(urlsplit(url).path.split("/")[-1]),"url":url.split("#")[0],"size":total,"sha256":sha,"source":"PyTorch cu126"})
CHUNK=16*1024**2
tasks=[]
for item in items:
 if (DEST/item["filename"]).exists():continue
 for start in range(0,item["size"],CHUNK):tasks.append((item,start,min(start+CHUNK,item["size"])-1))
def fetch(task):
 item,start,end=task;part=DEST/(item["filename"]+".part-"+str(start))
 if part.exists() and part.stat().st_size==end-start+1:return
 for attempt in range(4):
  try:
   sep="&" if "?" in item["url"] else "?"
   with requests.get(item["url"]+sep+f"part_start={start}",headers={"Range":f"bytes={start}-{end}"},stream=True,timeout=(10,80)) as r:
    r.raise_for_status()
    if r.status_code!=206 or r.headers.get("Content-Range")!=f"bytes {start}-{end}/{item['size']}":raise ValueError("Invalid wheel byte range")
    temp=part.with_name(part.name+".tmp");count=0
    with temp.open("wb") as f:
     for b in r.iter_content(1024**2):
      count+=len(b)
      if count>end-start+1:raise ValueError("Oversized wheel range")
      f.write(b)
    if count!=end-start+1:raise ValueError("Incomplete wheel range")
    temp.replace(part);return
  except (requests.RequestException,ValueError):
   if attempt==3:raise
   time.sleep(attempt+1)
print("Official runtime wheels",len(items),"ranges",len(tasks),flush=True)
with ThreadPoolExecutor(max_workers=24) as p:
 futures=[p.submit(fetch,t) for t in tasks]
 for i,f in enumerate(as_completed(futures),1):
  f.result()
  if i%32==0:print("Runtime ranges",i,"/",len(tasks),flush=True)
for item in items:
 target=DEST/item["filename"]
 if not target.exists():
  temp=target.with_name(target.name+".assembling")
  with temp.open("wb") as f:
   for start in range(0,item["size"],CHUNK):
    with (DEST/(item["filename"]+".part-"+str(start))).open("rb") as inp:
     for b in iter(lambda:inp.read(8*1024**2),b""):f.write(b)
  temp.replace(target)
 h=hashlib.sha256()
 with target.open("rb") as inp:
  for b in iter(lambda:inp.read(8*1024**2),b""):h.update(b)
 if h.hexdigest()!=item["sha256"] or target.stat().st_size!=item["size"]:raise ValueError("Runtime wheel hash mismatch")
 for start in range(0,item["size"],CHUNK):(DEST/(item["filename"]+".part-"+str(start))).unlink(missing_ok=True)
 print("Verified",item["filename"],flush=True)
# URLs are public; credentials are never used.
(BASE/"reports/sam3-runtime-wheels.json").write_text(json.dumps(items,indent=2))
print("RUNTIME WHEELS VERIFIED",flush=True)
