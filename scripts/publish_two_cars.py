from pathlib import Path
import json,html
p=Path("runs/two_cars_comparison");report=json.loads((p/"report.json").read_text())
def video(path):return f'<video controls muted loop preload="metadata" src="../{path}"></video>'
def pair(title,paths):
 return '<h2>'+title+'</h2><div class="grid">'+''.join(f'<article><h3>Step {st}</h3>{video(path)}</article>' for st,path in zip([20,50],paths))+'</div>'
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>两辆红车：Codex Agent 迭代与粗白模</title>
<style>body{background:#131925;color:#eee;font:17px system-ui;max-width:1500px;margin:30px auto;padding:22px;line-height:1.65}a{color:#8cf}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}video,img{width:100%;display:block;background:#222}article{min-width:0}h1{font-size:30px}h2{margin-top:40px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #465068;padding:10px;text-align:left}.note{padding:18px;border-left:4px solid #eeb44a;background:#202938}code{overflow-wrap:anywhere}li{margin:9px 0}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
<h1>两辆红车：Codex Agent 迭代与粗白模</h1>
<p class="note"><b>已完成真实工具执行及两次 Agent 方案修正。</b>当前对话的 Codex 担任 Agent，检查 RGB／深度／SAM候选，提出方案、调用工具、检查误差并决定接受或撤回。没有调用独立的外部 VLM API。结果仍是粗三维候选，未验证真实相机与物理几何。</p>
<p>输入为之前的 <code>wan_joint</code> 两车案例：近车横穿前景后向右离场，远车从其后显露。沿用同一50步采样内的 Step20／50 干净RGB估计，各自独立运行49帧VGGT与SAM3，未交换两阶段掩码或数值几何。原prompt只写一辆车，实体判断以画面中的两辆车为准。</p>
<h2>1 · 身份、遮挡和显露</h2><p>洋红：近车；青色：远车。近车SAM可见帧0–15；远车Step20为10–48，Step50为11–48。两辆车保持不同身份，没有把显露过程解释成同一车缩小。地面、立柱、左右书架分别建模，书架没有合并成一面连续背景墙。</p>
'''
page+=pair("分割跟踪：完整49帧",[f"two_cars_s{st}_reviewed/masks.mp4" for st in [20,50]])
page+='<h2>2 · 两车同时可见时的全局绕看</h2><p>冻结第12帧（0.75秒），只转观察视角。近车为粗盒体，远车为车身＋驾驶室＋四个车轮。所有场景几何保留，没有RGB背景贴图或远深度截断。</p>'
page+=pair("第12帧：两车空间关系",[f"two_cars_s{st}_overlap/orbit_both_cars.mp4" for st in [20,50]])
page+='<div class="grid">'+''.join(f'<article><h3>Step{st} 两车三视角</h3><img src="../two_cars_s{st}_overlap/both_cars.jpg"></article>' for st in [20,50])+'</div>'
page+=pair("3 · 最终原相机视角：RGB / 预测深度 / 白模 / 渲染深度",[f"two_cars_s{st}_delivery_r2/comparison.mp4" for st in [20,50]])
page+=pair("4 · 固定全局视角：播放物体运动",[f"two_cars_s{st}_global_final/global_fixed.mp4" for st in [20,50]])
page+=pair("两车区域的全局近看：只调整取景范围",[f"two_cars_s{st}_global_final/global_focus.mp4" for st in [20,50]])
page+='<p class="note">重建只覆盖有观测支持的运动区间：全局视频中近车在15帧后消失、远车从10／11帧出现，表示该时段以外的轨迹尚未重建，不能解释为物体在世界中被创建或销毁。遮挡段的完整轨迹仍是待补全信息。</p>'
page+=pair("5 · 全局透明线框、相机视锥和两车轨迹",[f"two_cars_s{st}_global_final/global_wireframe.mp4" for st in [20,50]])
page+='<p>蓝色：固定采用的VGGT相机；洋红／青色：两车。没有独立验证这些相机估计，也没有优化相机。</p>'
page+='<h2>6 · Agent 具体迭代了什么</h2><ol><li><b>R0：</b>两车各一个共享盒体，分别估计可见时段内的逐帧位姿；拟合地面、立柱与左右书架。</li><li><b>R1：</b>查看白模及误差后，为两车提出车身、驾驶室和四轮方案；共享形状重新拟合。轮胎和背侧对称性属于先验。</li><li><b>检查并撤回：</b>近车长期被边界裁切，完整车体方案使近车轮廓与深度拟合退步；拒绝近车R1。远车预测深度拟合改善，保留候选。</li><li><b>R2（当前采用）：</b>恢复并冻结R0近车与所有背景，只重拟合远车多部件方案。已验证冻结实体的形状与位姿逐项保持一致；远车轮廓和深度指标均改善。</li></ol>'
page+='<table><tr><th>阶段</th><th>轮次</th><th>近车轮廓IoU ↑</th><th>远车轮廓IoU ↑</th><th>远车深度误差 ↓</th></tr>'
for st in [20,50]:
 for rd in [0,1,2]:
  rr=report["stages"][str(st)]["rounds"][str(rd)]
  page+=f'<tr><td>Step{st}</td><td>R{rd}'+(' ✓ 当前采用' if rd==2 else '')+'</td>'+''.join(f'<td>{v*100:.1f}%</td>' for v in [rr["near_truck"]["mean_visible_frame_iou"],rr["far_truck"]["mean_visible_frame_iou"],rr["far_truck"]["mean_depth_error_on_target"]])+'</tr>'
page+='</table><p>IoU只在对应SAM实体有目标像素的帧上平均；深度误差为目标区域上的截断相对误差并惩罚渲染缺失，可能由其他实体的表面覆盖目标像素。指标相对于预测深度和SAM掩码；各帧都参与拟合，不是独立真实3D准确率。</p>'
page+=pair("被撤回的R1：两车都强行采用多部件",[f"two_cars_s{st}_delivery_r1/comparison.mp4" for st in [20,50]])
page+=pair("初始R0：两车都是盒体",[f"two_cars_s{st}_delivery_r0/comparison.mp4" for st in [20,50]])
page+=pair("7 · 最终深度残差：品红为缺失几何",[f"two_cars_s{st}_delivery_r2/residual.mp4" for st in [20,50]])
page+='<h2>8 · 当前边界</h2><ul><li>两车身份、局部遮挡与不同深度层次已有独立表达；重叠投影的检查点中近车几何在前，但没有完整的amodal真值。</li><li>近车仍是粗盒，长期裁切导致完整车长与形状不确定；远车隐藏轮胎属于对称性先验。</li><li>窗户、天花板和桌子尚未建模。全图几何覆盖约73–74%；没有用原RGB填掉缺失区域。</li><li>书架形状仍粗糙，可能存在深度形变；相机和物体轨迹均没有物理真值验证。Step50不保证所有几何指标优于Step20。</li><li>本次复用已有RGB读出，未重新生成视频，未测提前终止采样的计算收益。</li></ul>'
page+='<h2>9 · 可追溯产物</h2><ul>'
for st in [20,50]:
 page+=f'<li>Step{st}：<a href="../two_cars_s{st}_fit_r2/scene.json">最终动态场景JSON</a> · <a href="../two_cars_s{st}_overlap/both_cars_frame12.obj">两车同时可见的OBJ</a> · <a href="../two_cars_s{st}_delivery_r2/trajectories.json">相机／物体轨迹</a> · <a href="../two_cars_s{st}_hypothesis/hypothesis.json">Agent实体方案</a> · <a href="../two_cars_s{st}_reviewed/review.json">身份绑定</a> · <a href="../two_cars_s{st}_revision1_request/request.json">R1方案</a> · <a href="../two_cars_s{st}_revision2_request/request.json">R2撤回／局部重拟合决策</a></li>'
page+='</ul><p><a href="report.json">完整实验和诊断报告</a> · <a href="../stage20_50_comparison/index.html">上一单车案例</a> · <a href="../project_docs/index.html">项目计划与框架</a></p></html>'
(p/"index.html").write_text(page)
print(p/"index.html")
