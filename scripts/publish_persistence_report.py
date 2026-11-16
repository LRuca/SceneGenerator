"""Reporting only; frozen algorithm sources and numerical policy remain unchanged."""
from pathlib import Path
import json,html,shutil
import numpy as np
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
p=Path("runs/persistence_v1_report");r=json.loads((p/"report.json").read_text());summary=r["summary"]
labels=[];raw=[];cont=[];linear=[];held=[]
for record in r["records"]:
 for name,e in record["entities"].items():
  labels.append(record["case"].split("_")[0]+f' S{record["step"]} '+name)
  raw.append(e["observed_iou_raw"]);cont.append(e["observed_iou_persistent"])
  linear.append(e["withheld_iou_linear_baseline"]);held.append(e["withheld_iou_persistent"])
x=np.arange(len(labels));fig,axes=plt.subplots(1,2,figsize=(14,5))
for ax,a,b,title,l1,l2 in [(axes[0],raw,cont,"Observed masks: smoothness costs evidence fit","raw fit","persistent"),(axes[1],linear,held,"Withheld dynamic mask spans: compare interpolation","linear baseline","persistent")]:
 ax.bar(x-.18,a,.36,label=l1,color="#7d90ac");ax.bar(x+.18,b,.36,label=l2,color="#21b5c7")
 ax.set_xticks(x,labels,rotation=35,ha="right");ax.set_ylim(0,1);ax.set_ylabel("Visible-mask IoU");ax.set_title(title);ax.legend()
fig.tight_layout();fig.savefig(p/"iou_comparison.jpg",dpi=140);plt.close(fig)
def video(path):return f'<video controls muted loop preload="metadata" src="../{path}"></video>'
def pair(title,paths):
 return '<h3>'+title+'</h3><div class="grid">'+''.join(f'<article><h4>Step {st}</h4>{video(path)}</article>' for st,path in zip([20,50],paths))+'</div>'
style='''<style>body{background:#131925;color:#eee;font:17px system-ui;max-width:1500px;margin:30px auto;padding:22px;line-height:1.65}a{color:#8cf}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}video,img{width:100%;display:block;background:#222}article{min-width:0}h1{font-size:30px}h2{margin-top:40px}table{border-collapse:collapse;width:100%;font-size:15px}td,th{border:1px solid #465068;padding:8px;text-align:left}.note{padding:18px;border-left:4px solid #eeb44a;background:#202938}code,pre{white-space:pre-wrap;overflow-wrap:anywhere}li{margin:9px 0}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>'''
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>持续存在与连续运动：冻结流程验证</title>'+style+'''<h1>持续存在＋连续运动＋观测／推断分离</h1>
<p class="note"><b>第一版通用机制与冻结验证已完成；验证没有证明它普遍提高重建准确性。</b>实体不再因掩码缺失而被删除，轨迹抖动显著下降，但新案例的画面拟合与普通线性插值对照暴露了明显退步。以下保留全部四组结果，没有根据结果逐例改参数。</p>
<h2>1 · 实际新增的系统能力</h2>
<ul><li><b>存在 ≠ 可见 ≠ 有观测：</b>exists 记录镜头内实体生命周期；observation_available／used 记录观测与使用情况；visibility_state 记录粗几何估计的遮挡／画外／未检出。</li>
<li><b>连续运动：</b>共享刚体形状、稳健自然三次样条位置／朝向；隐藏内部区间插值，两端以连续速度延续，衔接满足C2。明确地面支撑关系时，最低表面保持接触。没有观测不再切断世界中的实体。</li>
<li><b>证据与推断分开：</b>state_source 标出观测支持、插值或外推；位置／朝向不确定性代理随远离证据增大。代理未校准，不是置信概率。</li>
<li><b>相机也分开：</b>原VGGT深度、K和c2w观测包完整保留；输出另存连续相机估计。投影误差变化可能来自物体与相机两方面，没有伪称几何观测未变就等于输出相机未变。</li>
<li><b>物理诊断：</b>检查地面接触和粗几何包围体相交；发生冲突会报告。本版没有碰撞求解器，也没有自动推理完整交通规则。</li></ul>
<p>没有微调Wan／VGGT／SAM3；这些是可复用的状态与优化代码。Agent仍由当前对话的Codex担任，只在拟合前做一次实体身份／静动态／支撑判断。</p>
<h2>2 · 如何避免再次成为逐样本修图</h2>
<p>先登记两个新提示词与种子（173／829），生成后不看画面；在旧案例、合成数据及23项测试通过后冻结算法源代码、参数、Agent协议和评价方法，随后才检查新输入。两例各取Step20／50，共四组；统一一盒体／实体，未加入车轮或其他样本专用模板。运行前后均校验冻结哈希。</p>
<p><a href="freeze.json">冻结记录</a> · <a href="policy.json">固定配置</a> · <a href="agent_protocol.html">Agent协议</a> · <a href="heldout_inputs.json">预登记提示词与种子</a> · <a href="frozen_source.zip">冻结源码包</a></p>
<h2>3 · 新案例的主要结果</h2>
'''
page+=f'''<ul><li>2个新案例 × 2个扩散阶段，共6条动态轨迹。所有实体在整个镜头保留存在状态，隐藏时段明确标为推断。</li>
<li>平移／角运动加速度抖动指标的中位降幅分别为 {summary["median_translation_acceleration_reduction_fraction"]*100:.1f}%／{summary["median_angular_acceleration_reduction_fraction"]*100:.1f}%。这是平滑效果，不能单独证明运动正确。</li>
<li>观测帧轮廓IoU的轨迹均值：{summary["mean_observed_iou_raw"]*100:.1f}% → {summary["mean_observed_iou_persistent"]*100:.1f}%，<b>下降</b>。</li>
<li>留出片段IoU：普通线性插值均值{summary["mean_withheld_iou_linear"]*100:.1f}%，连续机制{summary["mean_withheld_iou_persistent"]*100:.1f}%；仅{summary["holdout_beats_linear_count"]}/6条轨迹超过线性插值。<b>当前平滑／支撑／相机策略整体没有胜过更简单的对照。</b></li>
<li>四组中一组检测到包围体冲突。缺失几何还会使冲突检测漏报。</li></ul>'''
page+='<img src="iou_comparison.jpg"><table><tr><th>案例／阶段／实体</th><th>观测IoU：原→连续</th><th>留出IoU：线性→连续</th><th>观测／存在帧数</th><th>平移加速度剩余比例</th></tr>'
for record in r["records"]:
 for eid,e in record["entities"].items():
  page+=f'<tr><td>{record["case"]} S{record["step"]}<br>{eid}</td><td>{e["observed_iou_raw"]*100:.1f}% → {e["observed_iou_persistent"]*100:.1f}%</td><td>{e["withheld_iou_linear_baseline"]*100:.1f}% → {e["withheld_iou_persistent"]*100:.1f}%</td><td>{e["observed_count"]} / {e["existence_count"]}</td><td>{e["position_acceleration_ratio"]*100:.2f}%</td></tr>'
page+='</table><p>数值在各轨迹内比较；世界尺度任意，不跨场景比较绝对加速度。IoU基于SAM掩码而非人工真值。留出测试仅在下游初始化／几何拟合前移除连续目标掩码，VGGT仍使用完整RGB视频，因此不是端到端视觉留出，也不是未来预测。</p>'
for case,title in [("warehouse_crossing","仓库：红色配送车、蓝色卡车与立柱"),("courtyard_occlusion","院子：叉车、停放车辆与未重建的亭子")]:
 page+=f'<h2>4 · {title}</h2>'
 if case=="warehouse_crossing":page+='<p>红车只有30帧SAM观测，系统保留49帧存在状态；其余19帧为有标记的外推。静态货架区域仍是固定粗盒体，语义车头方向未专门求解。</p>'
 else:page+='<p class="note">生成失败：叉车从亭子前经过，没有形成预期的后方遮挡；未重新抽样。检测失败：两阶段的亭子SAM查询均为空，没有重试提示或手工补掩码。Step20叉车仅有32帧观测，17帧为推断；Step20检测到叉车与停放车辆的粗包围体相交22次。本例不能当作完整遮挡场景重建成功。</p>'
 page+=pair("RGB／原始观测门控白模／连续白模与证据颜色",[f"pv1_{case}_s{st}_experiment/view/source_evidence.mp4" for st in [20,50]])
 page+='<p>青色：观测支持的拟合状态；橙色：推断状态；灰色：静态实体。青色也不是测量真值。没有掩码但模型预测可见的帧会记录为visible_without_detection，可能是漏检，也可能是错误补全。</p>'
 page+=pair("同一固定全局视角：旧门控／持续存在",[f"pv1_{case}_s{st}_experiment/view/global_evidence.mp4" for st in [20,50]])
 page+='<div class="grid">'+''.join(f'<article><h4>Step{st} 轨迹、存在／观测与不确定性</h4><img src="../pv1_{case}_s{st}_experiment/view/state_curves.jpg"></article>' for st in [20,50])+'</div>'
 page+=pair("留出区间后的轨迹补全：原RGB／深度／白模／渲染深度",[f"pv1_{case}_s{st}_experiment/held_delivery/comparison.mp4" for st in [20,50]])
 page+=pair("相同留出区间的普通线性插值对照",[f"pv1_{case}_s{st}_experiment/held_linear_delivery/comparison.mp4" for st in [20,50]])
 page+='<ul>'
 for st in [20,50]:
  root=f"pv1_{case}_s{st}_experiment"
  page+=f'<li>Step{st}：<a href="../{root}/case_report.json">实验报告</a> · <a href="../{root}/persistent/scene.json">连续世界状态</a> · <a href="../{root}/view/states.json">逐帧存在／证据／可见性</a> · <a href="../{root}/persistent/report.json">连续性与接触诊断</a> · <a href="../{root}/physical_conflicts.json">物理冲突</a> · <a href="../pv1_{case}_s{st}_locked_plan/plan.json">拟合前锁定的Agent方案</a></li>'
 page+='</ul>'
page+='''<h2>5 · 有真值的合成验证</h2>
<p>已知共享形状、相机与轨迹，加入噪声并隐藏开头、中间、结尾的目标观测；连续机制只使用保留的样本。此部分可检验插值／外推、连续性和地面接触，但复杂度远低于电影画面。</p>'''
bench=json.loads(Path("runs/persistence_synthetic_benchmark/benchmark.json").read_text())
page+=f'<p>噪声位置RMSE={bench["raw_position_RMSE"]:.4f}；连续估计的观测时段RMSE={bench["persistent_observed_position_RMSE"]:.4f}；隐藏时段RMSE={bench["persistent_hidden_position_RMSE"]:.4f}（合成世界单位）。<a href="../persistence_synthetic_benchmark/benchmark.json">真值与误差记录</a></p>'
page+=video("persistence_synthetic_benchmark/view/global_evidence.mp4")
page+='''<h2>6 · 现在可以得出的结论</h2>
<p>已经从“只在被检测时渲染的几何列表”推进为“有持续实体、连续状态、观测证据与推断标签的粗世界”。但当前运动先验、地面约束和独立相机平滑会与图像证据冲突；参数冻结后，这一缺点在新案例中暴露出来。下一轮应把投影一致性纳入连续轨迹联合优化，并在不确定隐藏状态存在冲突时保留多个候选，而不是继续给当前案例补细节。</p>
<p>本轮到此保留冻结结果，不根据这些失败修改测试案例或数值参数。没有宣称完整物理规律、碰撞自由、校准概率或真实电影泛化。</p>
<p><a href="report.json">完整汇总</a> · <a href="completion_audit.json">需求与证据审计</a> · <a href="test_log.txt">23项测试记录</a></p></html>'''
(p/"index.html").write_text(page)
for src,name in [("experiments/persistence_v1/FREEZE.json","freeze.json"),("experiments/persistence_v1/policy.json","policy.json"),("experiments/persistence_v1/heldout_inputs.json","heldout_inputs.json"),("logs/persistence-tests-locked.log","test_log.txt")]:
 shutil.copyfile(src,p/name)
protocol=Path("experiments/persistence_v1/AGENT_PROTOCOL.md").read_text()
(p/"agent_protocol.html").write_text('<meta charset="utf-8">'+style+'<pre>'+html.escape(protocol)+'</pre>')
print(p/"index.html")
