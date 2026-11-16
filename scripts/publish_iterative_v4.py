"""Publish immutable local review with positive and negative experiment evidence."""
import argparse,html,json,shutil
from pathlib import Path
from scene_baseline.storage import BASE,ROOT,new_run,read_json,write_json,reference
from scene_baseline.quality_v4 import evaluate
def run(out):
 p=new_run(out);rows=[];views=[]
 for stage in [20,50]:
  court=BASE/"runs/v4_courtyard_r1"
  for variant in ["baseline","visible_depth","visible_soft_containment_free"]:
   path=court/f"step{stage}_{variant}/scene.json"
   row=evaluate(path);row.update(case="courtyard",stage=stage,variant=variant)
   rows.append(row)
  views.append((f"双车 Step{stage}：等预算候选",court/f"view_step{stage}_camera",f"court{stage}"))
  human=BASE/f"runs/v4_person_s{stage}_room_revision_r1"
  for variant in ["rigid_tube","deformable_tube"]:
   row=evaluate(human/variant/"scene.json");row.update(case="person",stage=stage,variant=variant);rows.append(row)
  views.append((f"人物 Step{stage}：刚性粗柱体 / 受限弯曲",human/"review_camera",f"person{stage}"))
  shape=BASE/f"runs/v4_courtyard_s{stage}_shape_revision_r1"
  if (shape/"review/report.json").exists():
   row=evaluate(shape/"fit/scene.json");row.update(case="courtyard",stage=stage,variant="extra160_shared_shape_revision");rows.append(row)
   views.append((f"双车 Step{stage}：额外160步共享形状修订（单独实验）",shape/"review",f"shape{stage}"))
 views.extend([("受控形变：已知几何、两帧 mask 缺失",BASE/"runs/v4_cage_controlled_r1/view_camera","known_cage")])
 for case in ["partial_occlusion","track_outliers"]:
  views.append(("受控挑战："+case,BASE/f"runs/v4_controlled_challenges_r1/{case}/view_camera",case))
 for title,path,name in views:
  if not (path/"index.html").exists():raise FileNotFoundError(path)
  (p/name).symlink_to(path,target_is_directory=True)
 write_json(p/"observed_metrics.json",{"rows":rows,"metric_definition":"foreground coverage requires correct entity label; missed entity pixels penalized as 1; real depth/masks are predictions, not 3D ground truth"})
 for name,source in [("controlled_initial","runs/v4_controlled_r1"),("controlled_combinations","runs/v4_controlled_r2"),("controlled_challenges","runs/v4_controlled_challenges_r1")]:
  write_json(p/(name+".json"),read_json(BASE/source/"report.json"))
 snapshot=p/"source_snapshot";snapshot.mkdir();refs=[]
 names=[*BASE.glob("scene_baseline/*v4.py"),*BASE.glob("scripts/*v4.py")]
 names += [BASE/"scene_baseline"/x for x in ["scene.py","geometry.py","delivery.py","global_views.py","physical_audit.py","recovery_view_v2.py","visibility_view_v3.py"]]
 for src in names:
  dest=snapshot/src.relative_to(BASE);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);refs.append(reference(dest))
 write_json(snapshot/"manifest.json",{"files":refs,"note":"final source snapshot; original R1 source snapshot retained separately; no weights/private dependency source"})
 evidence={}
 for name in ["v4_regression_final_r1","v4_recovery_final_r1","v4_visible_final_r1"]:
  src=BASE/"runs"/(name+".log")
  if not src.exists():raise FileNotFoundError(src)
  shutil.copyfile(src,p/src.name);evidence[name]=reference(src)
 scope={"status":"experimental_results_for_review_not_default_promotion",
  "five_directions":{"fitter":"R1 single terms, R2 combinations, controlled actual occlusion and bad tracks, independent court20/50",
   "representation":"shared rigid proxies and bounded tube bending; numpy/torch radial equivalence tested; radial fitting explicitly unsupported",
   "priority":"narrative and geometric duties independent; executable budgets used by person fits; structural background retained; revision gates",
   "agent":"current-chat Codex; input-bound identity policies and diagnostic requests; actual wall representation revision and exploratory shape revision; no external VLM",
   "validation":"source/global/camera videos, camera numeric export, root/camera trajectories, observed metrics, known-geometry metrics, retained failures"},
  "limitations":["Real cases have no 3D ground truth. Predicted depth/masks can be jointly wrong.",
   "Equal-budget court Step20 silhouette can regress; do not promote all-on constraints.",
   "Fixed incorrect shape cannot be repaired by point containment alone.",
   "Single tube misses head and separated limbs; coarse motion only.",
   "Missing masks retain entities but inferred motion is unverified.",
   "Reference centroid convention and naive alternating optimizer were negative results.",
   "Conservative overlap audits are not a proof of physical collision or its absence.",
   "Additional shared-shape revision has additional budget and is not the sparse ablation.",
   "SAM remains auxiliary; automated diagnostic queue still requires current-chat agent decisions."],
  "tests":evidence,"source_snapshot":reference(snapshot/"manifest.json")}
 write_json(p/"evidence.json",scope)
 table=[]
 for row in rows:
  eid="forklift" if row["case"]=="courtyard" else "person";m=row["per_entity"][eid]
  table.append(f"<tr><td>{row['case']} {row['stage']}</td><td>{row['variant']}</td><td>{m['mask_iou']:.3f}</td><td>{m['foreground_coverage']:.3f}</td><td>{m['capped_depth_error_with_misses']:.3f}</td></tr>")
 cards=[]
 for title,path,name in views:
  cards.append(f'<section><h2>{html.escape(title)}</h2><p><a href="{name}/index.html">完整对照、轨迹和误差</a></p><video src="{name}/source_comparison.mp4" controls preload="metadata"></video><video src="{name}/global_comparison.mp4" controls preload="metadata"></video></section>')
 doc="""<!doctype html><meta charset="utf-8"><title>SceneGenerator V4 实验审阅</title><style>body{font:16px system-ui;background:#141b25;color:#e4edf5;max-width:1200px;margin:30px auto;padding:16px}a{color:#83caff}video{width:100%;max-height:650px}td,th{padding:8px;border-bottom:1px solid #526070}section{margin-top:40px}table{border-collapse:collapse}</style><h1>SceneGenerator V4：五项迭代实验</h1><p>原有拟合器的增量实验。Step20/50独立处理，固定各自VGGT相机；后期视频不监督前期。全部视频包含完整画幅，几何没有按深度裁切。全局视角绿色表示相机轨迹；数值相机与轨迹图见各完整页面。</p><p><strong>结果需审阅：</strong>可见深度项在受控几何上最有效；真实双车的收益存在取舍，暂不替换默认拟合器。人物仅验证受限弯曲粗柱体，无细节人体重建。Agent由当前对话Codex担任，未调用外部VLM。</p><h2>可见前景指标</h2><p>以下是对预测深度和辅助mask的一致性，不是真实三维精度。覆盖率只计算正确实例；漏建不能由背景填充抵消。带漏检惩罚的深度误差越低越好。</p><table><tr><th>输入</th><th>方案</th><th>mask IoU ↑</th><th>覆盖 ↑</th><th>深度含漏建惩罚 ↓</th></tr>"""+"".join(table)+"</table><p><a href='observed_metrics.json'>逐帧指标</a> · <a href='evidence.json'>五项证据与限制</a> · <a href='controlled_initial.json'>初始消融（保留失败）</a> · <a href='controlled_combinations.json'>轻量组合</a> · <a href='controlled_challenges.json'>遮挡与错误跟踪</a></p>"+"".join(cards)
 (p/"index.html").write_text(doc)
 print(p,flush=True)
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);run(ap.parse_args().out)
