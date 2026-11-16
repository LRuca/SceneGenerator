"""Frozen controlled-case configuration evaluated independently on Step20/50."""
from pathlib import Path
import contextlib,json,argparse
from scene_baseline.storage import ROOT,BASE,new_run,read_json,write_json,reference
from scene_baseline.sparse_fit_v4 import fit
def run(out):
 p=new_run(out);cfg=read_json(BASE/"experiments/iterative_v4/real_candidate.json")
 write_json(p/"configuration.json",cfg)
 rows=[]
 for stage in [20,50]:
  old=ROOT/f"sceneblocker/baseline/runs/pv2_courtyard_s{stage}_pipeline_r2"
  tracks=ROOT/f"sceneblocker/baseline/runs/pv2_courtyard_s{stage}_tracks_auto"
  for variant,opts in cfg["variants"].items():
   dest=p/f"step{stage}_{variant}"
   with (p/f"step{stage}_{variant}.log").open("x") as log,contextlib.redirect_stdout(log):
    rep=fit(old/"surfaces/scene.json",tracks/"manifest.json",tracks/"agent_review.json",dest,steps=cfg["steps"],options=opts)
   row={"stage":stage,"variant":variant,"scene":reference(dest/"scene.json"),
      "report":reference(dest/"report.json"),"point_error_px":rep["median_reprojection_px"],
      "outside_proxy_fraction":rep["outside_proxy_fraction"]}
   rows.append(row);print(json.dumps(row),flush=True)
 write_json(p/"report.json",{"rows":rows,"source_config":reference(BASE/"experiments/iterative_v4/real_candidate.json"),
   "status":"candidate_fits_complete_visualization_and_review_required","cross_stage_inputs":False})
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);a=ap.parse_args();run(a.out)
