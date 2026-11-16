"""Publish real Step20/50 coarse-model results without claiming geometry correctness."""
from pathlib import Path
import json,numpy as np
from scene_baseline.storage import reference
p=Path("runs/stage20_50_comparison");p.mkdir(exist_ok=False)
records={}
for st in [20,50]:
 m=json.load(open(f"runs/stage{st}_delivery_v2/metrics.json"))
 fit=json.load(open(f"runs/stage{st}_fit/report.json"))
 s=json.load(open(f"runs/stage{st}_fit/scene.json"))
 a=np.load(f"runs/stage{st}_geometry/geometry.npz")
 iou={eid:float(np.mean([v["entities"][eid]["visible_mask_iou"] for v in m["frames"] if v["entities"][eid]["visible_mask_iou"] is not None])) for eid in ["vehicle","ground","column","shelves"]}
 med=float(np.median(a["depth"]));poses=np.array(s["entities"][0]["entity_to_world"])
 records[str(st)]={"summary":m["summary"],"mean_mask_iou":iou,"fit_seconds":fit["seconds"],
  "scene":reference(f"runs/stage{st}_fit/scene.json"),
  "camera_end_displacement_over_median_depth":float(np.linalg.norm(a["c2w"][-1,:3,3]-a["c2w"][0,:3,3])/med),
  "vehicle_end_displacement_over_median_depth":float(np.linalg.norm(poses[-1,:3,3]-poses[0,:3,3])/med),
  "vehicle_end_rotation_deg":float(np.degrees(np.arccos(np.clip((np.trace(poses[-1,:3,:3]@poses[0,:3,:3].T)-1)/2,-1,1))))}
report={"status":"real_coarse_3d_candidate_completed_quality_not_accepted","stages":records,
 "inputs":"same Wan generated clip, seed42, 49 frames, 16fps, separate Step20/50 x0 RGB previews",
 "generation_budget":"readouts reused from existing 50-step run; not an early-stopped generation benchmark",
 "geometry":"per-stage independent same-pass VGGT depth/confidence/K/c2w, no cross-stage mask or geometry transfer",
 "hypothesis":"Codex visual review; no automated external VLM agent service",
 "segmentation":"SAM3 local ModelScope checkpoint; stationary shelves grouped after review; Step20 shelves masks missing frames0-9",
 "fit":"fixed predicted cameras; 4 shared primitives, one dynamic box; 200 Adam steps at64x64; shape, centers and yaw fit jointly",
 "observations":["Step50 has sharper RGB and more complete background segmentation",
 "both stages explain much of predicted depth but retain depth-elongated slab-like pillar",
 "vehicle is only one coarse box, hidden shape not recovered",
 "camera and vehicle paths are estimated rather than physically validated",
 "same-view depth/mask agreement cannot certify correct 3D geometry"],
 "scope":"all frames used for fitting; no independent heldout or metric scale evaluation"}
(p/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
def video(path):return f'<video controls muted loop preload="metadata" src="../{path}"></video>'
def pair(title,paths):
 return f'<section><h2>{title}</h2><div class="grid">'+''.join(f'<article><h3>Step {st}</h3>{video(path)}</article>' for st,path in zip([20,50],paths))+'</div></section>'
table='<table><tr><th>同源观测一致性指标</th><th>Step20</th><th>Step50</th></tr>'
for name,key in [("全图渲染覆盖率","mean_coverage"),("含缺失惩罚的截断相对深度误差 ↓","mean_capped_relative_error_with_misses")]:
 table+=f'<tr><td>{name}</td>'+''.join(f'<td>{records[str(st)]["summary"][key]*100:.1f}%</td>' for st in [20,50])+'</tr>'
table+='<tr><td>车辆可见轮廓 IoU ↑</td>'+''.join(f'<td>{records[str(st)]["mean_mask_iou"]["vehicle"]*100:.1f}%</td>' for st in [20,50])+'</tr></table>'
page='''<!doctype html><meta charset="utf-8"><title>Step20 / Step50：真实案例粗三维与全局视角</title>
<style>body{background:#131925;color:#eee;font:17px system-ui;max-width:1500px;margin:30px auto;padding:22px;line-height:1.65}a{color:#8cf}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}video,img{width:100%;display:block;background:#222}article{min-width:0}h1{font-size:30px}h2{margin-top:40px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #465068;padding:10px;text-align:left}.note{padding:18px;border-left:4px solid #eeb44a;background:#202938}code{overflow-wrap:anywhere}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
<h1>Step20 / Step50：真实案例粗三维与全局视角</h1>
<p class="note"><b>已产出真实案例的显式三维候选，但质量尚未验收通过。</b>两组均包含地面、书架背景、立柱和运动的车体盒；共享形状＋逐帧位姿＋VGGT相机轨迹。全局视角显示立柱被拉成长板、车辆缺乏细节，因此不能把流程完成等同于重建正确。</p>
<p>同一条 Wan 生成视频，49帧、16fps；分别读取已有50步采样中 Step20 与 Step50 对干净视频的估计，不重新生成。各自独立运行 VGGT 与 SAM3，未用后期帧补早期几何。此实验测试不同去噪阶段的读出质量，还不是提前停止采样的耗时实验。</p>
'''
page+=pair("1 · 全局固定视角：播放的是物体运动", [f"stage{st}_global_v2/global_fixed.mp4" for st in [20,50]])
page+=pair("2 · 全局绕看：冻结第24帧，只转观察视角", [f"stage{st}_global_v2/orbit_midtime.mp4" for st in [20,50]])
page+='<h2>3 · 透明线框与轨迹</h2><p>青色：VGGT相机位置与视锥；橙色：拟合车体轨迹。坐标经过同一个刚性变换对齐估计的地面朝上，三个轴等比例显示。透明面仅便于观察遮挡，不修改场景。</p>'
page+=pair("逐帧全局线框", [f"stage{st}_global_v2/global_wireframe.mp4" for st in [20,50]])
page+='<div class="grid">'+''.join(f'<article><h3>Step {st}</h3><img src="../stage{st}_global_v2/global_wireframe.jpg"></article>' for st in [20,50])+'</div>'
page+=pair("4 · 原相机下：RGB / 输入深度 / 白模 / 渲染深度",[f"stage{st}_delivery_v2/comparison.mp4" for st in [20,50]])
page+=pair("5 · 分割与身份检查",[f"stage{st}_reviewed/masks.mp4" for st in [20,50]])
page+='<p>红色车辆、绿色地面、青色立柱、黄色书架。Step20 书架在前10帧漏检，保留缺失；将后续静态书架分割合为一个粗背景块，并记录于review.json。未合并动态车辆身份。</p>'
page+=pair("6 · 重投影深度残差",[f"stage{st}_delivery_v2/residual.mp4" for st in [20,50]])
page+='<p>右侧为相对深度误差，色域0–0.5；品红表示没有几何覆盖。深度预览使用全段完整对数范围，没有裁掉中后景。</p>'
page+='<h2>7 · 如何解读</h2>'+table+'''<p>这些指标相对于同一次VGGT的预测深度和SAM3掩码；所有49帧都参与拟合。它们反映对输入的解释程度，不能当作真实3D或相机精度。Step50改善背景分割和深度解释，但车辆盒的轮廓IoU基本未变。</p>
<p>当前每个实体只有一个基本几何体，200次优化、64×64拟合分辨率；相机保持VGGT估计。车辆朝向变化由拟合读出，但平移/旋转的物理真实性未验证。立柱的深度拉伸暴露了深度融合、朝向与形状先验不足；清晰RGB本身无法消除该问题。</p>
<h2>8 · 可复查产物</h2><p>scene.json 含共享形状、静态/逐帧位姿及观测绑定；OBJ 为中间时刻实际模型，可导入3D软件。</p><ul>
'''
for st in [20,50]:
 page+=f'<li>Step{st}：<a href="../stage{st}_fit/scene.json">动态场景JSON</a> · <a href="../stage{st}_global_v2/scene_midtime.obj">中间帧OBJ</a> · <a href="../stage{st}_delivery_v2/trajectories.json">相机/物体轨迹</a> · <a href="../stage{st}_fit/report.json">拟合记录</a> · <a href="../stage{st}_reviewed/review.json">分割审查</a></li>'
page+='</ul><p><a href="report.json">本次报告JSON</a> · <a href="../project_docs/index.html">框架与计划</a></p>'
(p/"index.html").write_text(page)
print(p/"index.html")
