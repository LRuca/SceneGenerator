"""Local-only dashboard and reviewable plans. Exposes runs, never secrets or weights."""
from pathlib import Path
import json,html,shutil,re
from scene_baseline.storage import ROOT,BASE
p=BASE/"runs/project_docs";p.mkdir(exist_ok=True)
for name in ["PLAN.md","EXECUTION.md","RUNBOOK.md","docs/INTERFACES.md","CODE_MAP.md"]:
    source=BASE/name;dest=p/(source.stem+".html")
    dest.write_text('<meta charset="utf-8"><style>body{max-width:1100px;margin:40px auto;background:#131925;color:#eee;font:16px system-ui}pre{white-space:pre-wrap;line-height:1.7}a{color:#8cf}</style><a href="index.html">返回</a><pre>'+html.escape(source.read_text())+'</pre>')
summary={"root":str(BASE),"E0_environment":"passed","E1_contracts_and_tools":"passed_16_tests",
 "E2_same_pass_vggt":"passed_real_49_frames","E3_processing_tools":"passed_real_stage10",
 "E4_agent":"file_interface_tested_remote_provider_not_configured",
 "E4_sam3":"in_progress" if not (BASE/"reports/sam3-weight.json").exists() else "weights_verified_runtime_integration_not_tested",
 "E5_fitting":"joint_static_and_rigid_scene_fitter_passed_synthetic",
 "E6_render":"full_delivery_videos_depth_errors_trajectories_passed_synthetic",
 "E7_real_scene_baseline":"pending_masks_review_and_full_scene_fit",
 "E8_agent_correction_loop":"scene_revision_and_depth_adoption_tested_remote_agent_pending"}
if (BASE/"runs/stage10_sam3_vehicle/manifest.json").exists():
    summary["E4_sam3"]=json.loads((BASE/"runs/stage10_sam3_vehicle/manifest.json").read_text())["status"]
if (BASE/"runs/stage20_50_comparison/report.json").exists():
    summary.update({"E4_sam3":"real_step20_and_step50_49frame_tracking_completed_reviewed",
      "E5_fitting":"real_step20_and_step50_fixed_camera_shared_primitive_fit_completed",
      "E6_render":"real_source_global_orbit_wireframe_OBJ_and_trajectories_completed",
      "E7_real_scene_baseline":"coarse_3d_candidate_completed_quality_not_accepted_pillar_depth_elongation"})
if (BASE/"runs/two_cars_comparison/report.json").exists():
    summary.update({"E4_agent":"current_chat_Codex_agent_visual_review_tool_execution_and_two_revision_rounds_completed_no_external_VLM",
      "two_car_case":"real_step20_50_two_SAM_identities_R0_R1_R2_local_refit_and_global_views_completed_coarse_quality"})
(BASE/"reports/status.json").write_text(json.dumps(summary,indent=2))
(p/"status.json").write_text(json.dumps(summary,indent=2))
for name in ["environment.json","sam3-readiness.json","sam3-weight.json","tool-catalog.json"]:
    src=BASE/"reports"/name
    if src.exists():shutil.copyfile(src,p/name)
fit=json.loads((BASE/"runs/synthetic_smoke/fit.json").read_text())
text=f"""<!doctype html><meta charset="utf-8"><title>SceneBlocker baseline framework</title>
<style>body{{max-width:1200px;margin:35px auto;padding:20px;background:#131925;color:#eee;font:18px system-ui}}a{{color:#8cf}}video{{max-width:100%;width:850px}}img{{max-width:100%}}pre{{white-space:pre-wrap}}li{{margin:10px}}</style>
<h1>早期视频 → 粗白模：框架与功能验证</h1>
<p>以下为组件和环境测试。联合拟合与完整交付已在合成及真实案例跑通。真实Step20/50已完成分割、拟合与全局视角，但存在立柱拉伸等几何错误；外部agent服务未配置。</p>
<ul><li><a href="PLAN.html">整体计划与复用</a></li><li><a href="EXECUTION.html">执行阶段、下载与测试计划</a></li><li><a href="RUNBOOK.html">实际命令与产物位置</a></li><li><a href="INTERFACES.html">Agent／工具／几何／掩码接口</a></li></ul>
<h2>真实 Step10：49帧统一 VGGT 读出</h2><p>同一次特征计算产生深度、置信度、内参与相机位姿；约21秒，峰值显存21.4 GiB。相机是预测初始化。</p><video controls loop muted src="../stage10_geometry/rgb_depth.mp4"></video>
<h2>已测试的处理工具</h2><p>原图／双边去噪／锐化。只作为接口与处理分支测试；没有据此声称几何精度提高。</p><img src="../tool_comparison.jpg"><p><a href="tool-catalog.json">机器可读工具目录</a></p>
<h2>完整合成场景链路</h2><p>地面＋背景＋运动物体：深度与掩码初始化→联合拟合→白模与误差。<a href="../synthetic_end_to_end_delivery_v2/index.html">打开完整结果</a></p><video controls loop muted src="../synthetic_end_to_end_delivery_v2/comparison.mp4"></video><h2>合成几何组件检查</h2><p>已知相机与盒体深度，扰动位置后拟合。残差 MSE：{fit['initial_mse']:.6f} → {fit['final_mse']:.6f}。这是组件验证。</p><video controls loop muted src="../synthetic_smoke/synthetic_white.mp4"></video>
<h2>状态</h2><pre>{html.escape(json.dumps(summary,indent=2))}</pre>"""
if (BASE/"runs/stage10_sam3_vehicle/overlay.mp4").exists():text+='<h2>SAM3车辆跟踪功能测试</h2><video controls loop muted src="../stage10_sam3_vehicle/overlay.mp4"></video><p>初始主体假设由Codex提供，身份与掩码仍需审查。</p>'
if (BASE/"runs/stage20_50_comparison/index.html").exists():text='<a style="font-size:24px" href="../stage20_50_comparison/index.html">最新：Step20 / Step50 真实粗三维与全局视角</a>'+text
if (BASE/"runs/two_cars_comparison/index.html").exists():text='<a style="font-size:24px" href="../two_cars_comparison/index.html">最新：两车案例与Codex Agent迭代</a><br>'+text
(p/"index.html").write_text(text)
link=ROOT/"sceneblocker/pre-analysis/outputs/baseline"
if not link.exists():link.symlink_to(BASE/"runs",target_is_directory=True)
print(str(p/"index.html"))
