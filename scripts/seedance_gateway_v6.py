"""Documented Venus Seedance interface, explicit request/task files; credentials never logged."""
import argparse,json,os
from pathlib import Path
import requests
from scene_baseline.storage import inside,read_json,write_json
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--run",required=True);ap.add_argument("--base",default="http://v2.open.venus.oa.com/llmproxy");ap.add_argument("--query",action="store_true");a=ap.parse_args()
 run=inside(a.run);secret=inside(os.environ["SB_VENUS_SECRET_FILE"]).read_text().strip()
 headers={"Authorization":"Bearer "+secret,"Content-Type":"application/json; charset=utf-8"}
 try:
  if a.query:
   task=read_json(run/"task.json")["task_id"];r=requests.get(a.base+"/v2/video/query",headers=headers,params={"task_id":task},timeout=(8,60),allow_redirects=False)
  else:
   if (run/"task.json").exists():raise ValueError("Existing task; query it instead of duplicate submission")
   r=requests.post(a.base+"/v2/video/generate",headers=headers,json=read_json(run/"request.json"),timeout=(8,90),allow_redirects=False)
 except requests.RequestException as e:
  write_json(run/("query_error.json" if a.query else "submit_error.json"),{"base":a.base,"error_type":type(e).__name__,"task_submitted":"unknown_if_submission_timed_out" if not a.query else "existing","secret_logged":False})
  raise SystemExit("Gateway connection failed; credential omitted")
 try:d=r.json()
 except ValueError:
  write_json(run/("query_error.json" if a.query else "submit_error.json"),{"base":a.base,"http_status":r.status_code,"content_type":r.headers.get("Content-Type"),"body_bytes":len(r.content),"error":"non_JSON_response_no_task_id","secret_logged":False})
  raise SystemExit("Gateway returned non-JSON response; no task ID")
 if d.get("code") not in [0,None] or not isinstance(d.get("data"),dict):
  write_json(run/("query_error.json" if a.query else "submit_error.json"),{"http_status":r.status_code,"code":d.get("code"),"message":str(d.get("message"))[:500]})
  raise SystemExit("Gateway rejected request")
 data=d["data"];task=data.get("task_id")
 if not a.query:
  if not task:raise SystemExit("No task ID")
  write_json(run/"task.json",{"task_id":task,"model":"seedance-2.5","base":a.base})
 else:
  write_json(run/"status.json",{"task_id":task,"status":data.get("status"),"status_message":data.get("status_message"),"usage":data.get("usage")})
  if data.get("status")=="succeed":
   urls=data.get("response_attachment_urls") or [];url=next((u for u in urls if u.split("?")[0].lower().endswith(".mp4")),None)
   if not url:raise SystemExit("Success without MP4")
   video=requests.get(url,timeout=(10,120));video.raise_for_status();(run/"generated.mp4").write_bytes(video.content)
 print(json.dumps({"task_id":task,"status":data.get("status")}))
if __name__=="__main__":main()
