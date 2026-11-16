from pathlib import Path
import json
from scene_baseline.storage import read_json,write_json,reference
p=Path("runs/two_cars_semantic_comparison");records={}
for st in [20,50]:
 records[str(st)]={"heading":read_json(f"runs/two_cars_s{st}_heading_review/report.json"),
 "agent":read_json(f"runs/two_cars_s{st}_semantic_r3/agent_decision.json"),
 "final_metrics":read_json(f"runs/two_cars_s{st}_delivery_semantic/metrics.json")["summary"],
 "scene":reference(f"runs/two_cars_s{st}_semantic_final/scene.json")}
write_json(p/"report.json",{"task":"user-reported wrong small-car orientation and solid background block",
 "actor":"current conversation Codex agent","stages":records,
 "orientation_fix":"reviewed SAM wheel4 rear-same-side,5 front-same-side,6 front-opposite; contact rays intersect estimated ground; shared shape and 2D contact reprojection fit; semantic +X front/yaw locked",
 "background_fix":"reject cross-depth contamination in bookcase masks; front-plane/ground-contact initialization; eight-part upright/shelf proxy, no inferred solid hidden volume",
 "floor":"same plane extended in XZ under reconstructed footprint; unobserved extent is a prior",
 "evaluation":"old and new use same original SAM masks; fitting masks cleaned but not substituted into comparison",
 "limitations":["wheel correspondence and estimated ground/camera are not independent 3D truth",
 "books, rear panels and table contents unmodeled; bookshelf filled-mask agreement/overall coverage may decline",
 "right bookshelf Step20 has insufficient ground contacts; uses explicit front-plane prior",
 "near vehicle remains clipped coarse box; hidden trajectories not recovered"]})
def video(path):return f'<video controls muted loop preload="metadata" src="../{path}"></video>'
def pair(title,paths):
 return '<h2>'+title+'</h2><div class="grid">'+''.join(f'<article><h3>Step {st}</h3>{video(path)}</article>' for st,path in zip([20,50],paths))+'</div>'
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>两车修正：语义朝向与书架结构</title>
<style>body{background:#131925;color:#eee;font:17px system-ui;max-width:1500px;margin:30px auto;padding:22px;line-height:1.65}a{color:#8cf}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}video,img{width:100%;display:block;background:#222}article{min-width:0}h1{font-size:30px}h2{margin-top:40px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #465068;padding:10px;text-align:left}.note{padding:18px;border-left:4px solid #eeb44a;background:#202938}li{margin:9px 0}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
<h1>两车修正：小车语义朝向＋书架结构</h1>
<p class="note"><b>用户指出的两处问题成立。</b>原方案把可见深度点的PCA主轴当作车长轴；右前方背景则由整个书架区域的实心包围盒代替，深度污染又放大了体积。上一轮仅检查轮廓与深度一致性，没有充分校验语义朝向和部件结构。</p>
<h2>1 · 小车朝向：RGB部位对应 → 几何约束</h2><p>重新运行SAM3 wheel查询，并由Codex检查第16／32／48帧：候选4为同侧后轮、5为同侧前轮、6为异侧前轮。其他候选存在前景身份或语义切换，未用于姿态。前向定义为4→5，射线与估计地面相交后给出约23–30°的朝向修正。</p>
<p>左：原RGB和轮胎接触点；中：旧模型及+X车头箭头；右：修正模型及+X车头箭头。此视频为第16–48帧的同一像素裁剪，不重生成RGB。</p>
'''
page+=pair("朝向前后对比",[f"two_cars_s{st}_heading_review/heading_comparison.mp4" for st in [20,50]])
page+='<table><tr><th>输入阶段</th><th>远车IoU：旧→新</th><th>车轮接触点平均重投影误差：旧→新</th><th>相对旧模型的朝向变化中位数</th></tr>'
for st in [20,50]:
 rr=records[str(st)];h=rr["heading"];err=h["wheel_contact_reprojection_mean_px"];yaw=rr["agent"]["orientation"]["median_change_deg"]
 page+=f'<tr><td>Step{st}</td><td>{h["old_visible_iou"]["far_truck"]*100:.1f}% → {h["new_visible_iou"]["far_truck"]*100:.1f}%</td><td>{err["old_best_side_mapping"]:.1f}px → {err["new"]:.1f}px</td><td>{yaw:.1f}°</td></tr>'
page+='</table><p>同一套原始SAM整车掩码用于前后评估，没有换成清理后的掩码抬高IoU。车轮点参与了拟合，因此点误差表示拟合一致性；它与IoU均不能当作独立真实三维精度。相机与地面错误仍可能影响朝向。近车几何保持原样。</p>'
page+='<h2>2 · 背景为什么是一整块：用整体盒体代替了书架</h2><p>原书架整体盒体没有立柱、层板和空隙；把某些混入的近处深度也纳入包围盒后，得到沿纵深拉长的实心几何。现改为薄的前表面／地面接触初始化，再由两个竖框和六个层板表达。厚度属于有记录的形状先验，不再将未知背面解释成巨大实体。</p><p>以下前后使用<b>相同全局观察相机</b>。金色为左书架、紫色为右书架、青色为远车；画面左右随观察视角变化，不等于原RGB中的左右。</p>'
for st in [20,50]:page+=f'<h3>Step{st}：同一全局观察相机，旧／新</h3><img src="global_before_after_s{st}.jpg">'
page+='<p class="note"><b>书架结构仍是部分模型。</b>书本、背板与桌子尚未完整恢复；空架结构减少了原先填满的区域，因此背景轮廓覆盖和全图覆盖率下降，不能称为完整背景重建。地面仅延展同一个估计平面到场景下方，延展区域是先验，不是新增测量。</p>'
page+=pair("3 · 修正后的两车全局绕看：冻结第12帧",[f"two_cars_s{st}_overlap_semantic/orbit_both_cars.mp4" for st in [20,50]])
page+=pair("4 · 修正后的原相机视角：RGB / 深度 / 白模 / 渲染深度",[f"two_cars_s{st}_delivery_semantic/comparison.mp4" for st in [20,50]])
page+=pair("5 · 固定全局视角：物体运动",[f"two_cars_s{st}_global_semantic/global_fixed.mp4" for st in [20,50]])
page+=pair("全局线框、相机与独立物体轨迹",[f"two_cars_s{st}_global_semantic/global_wireframe.mp4" for st in [20,50]])
page+='<p>蓝色相机、洋红近车、青色远车。未观测时段仍没有完整轨迹；全局出现／消失是观测门控，不能解释为物理创建／销毁。</p>'
page+=pair("6 · 深度残差：品红为缺失几何",[f"two_cars_s{st}_delivery_semantic/residual.mp4" for st in [20,50]])
page+='<h2>7 · 失败与先验都保留记录</h2><ul><li>桌子探测：Step20的SAM查询没有返回掩码；Step50仅得到部分碎片，均未作为可靠桌体采用，没有用另一阶段补齐。</li><li>书架拟合排除了跨深度污染像素，但最终评估仍使用原始掩码。排除数量、深度带、表面锚帧与厚度先验均记录。</li><li>Step20右书架缺少可靠地面接触线，采用显式的朝向相机的竖直前平面假设；仍需进一步验证。</li><li>相机固定为同次VGGT预测；没有独立真实姿态／几何真值，也没有新的3D模型训练。</li></ul><h2>8 · 产物</h2><ul>'
for st in [20,50]:
 page+=f'<li>Step{st}：<a href="../two_cars_s{st}_semantic_final/scene.json">最终场景JSON</a> · <a href="../two_cars_s{st}_overlap_semantic/both_cars_frame12.obj">两车同时可见OBJ</a> · <a href="../two_cars_s{st}_semantic_r3/agent_decision.json">Agent诊断与先验</a> · <a href="../two_cars_s{st}_heading_review/report.json">姿态诊断</a> · <a href="../two_cars_s{st}_delivery_semantic/metrics.json">最终同源误差</a></li>'
page+='</ul><p><a href="report.json">完整报告</a> · <a href="../two_cars_comparison/index.html">上一版R0–R2及失败方案</a></p></html>'
(p/"index.html").write_text(page)
old=Path("runs/two_cars_comparison/index.html")
text=old.read_text();banner='<div style="padding:20px;background:#4a351b;font-size:22px"><a href="../two_cars_semantic_comparison/index.html">最新修正：小车语义朝向与书架结构。此页保留为旧版R0–R2。</a></div>'
old.write_text(text.replace('<h1>',banner+'<h1>',1))
p=Path("RUNBOOK.md")
with p.open("a") as f:f.write("""
## User-reviewed semantic correction
runs/two_cars_semantic_comparison/index.html compares old/new heading and bookcase topology.
SAM wheel4=rear same-side,5=front same-side,6=front opposite were visually reviewed.
Ray/ground intersections initialize yaw; --entities far_truck shelf_left shelf_right,
per-entity fit_constraints.lock_yaw and 2D contact landmark term preserve semantic front.
Landmarks are serialized in fitted entity scale. Delivery supports
--evaluation-masks-manifest to compare against the same ORIGINAL masks after fit-mask filtering.
Bookcases use two uprights+six boards, thin bounded depth; contents unmodeled.
Final geometry runs/two_cars_s{20,50}_semantic_final/scene.json extends only the ground
patch XZ footprint beneath geometry; same normal/height, unobserved extension is a prior.
Full coverage worsens because books/occluders remain missing; vehicle silhouette improves.
Table SAM query failed/fragmented and was not silently replaced.
""")
print("Published",Path("runs/two_cars_semantic_comparison/index.html"))
