import sys,json,importlib.util,subprocess,platform
from pathlib import Path
from scene_baseline.storage import BASE,write_json
import torch
from scene_baseline.geometry import REF
modules=["torch","numpy","scipy","cv2","PIL","imageio","skimage","trimesh","sam3"]
d={"python":sys.version,"executable":sys.executable,"platform":platform.platform(),
 "modules":{m:bool(importlib.util.find_spec(m)) for m in modules},
 "torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
 "reference_source":str(REF),"reference_commit":"4a0fa059e24f43a2e0988daed9605b91214be3c7",
 "geometry_environment":"reused, no pip mutations by baseline"}
p=BASE/"reports/environment.json"
if p.exists():p.unlink()
write_json(p,d)
r=subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True,check=True)
(BASE/"reports/geometry-pip-freeze.txt").write_text(r.stdout)
print(json.dumps(d,indent=2))
