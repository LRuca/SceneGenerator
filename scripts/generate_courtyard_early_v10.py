import sys,time,json,types,math
from pathlib import Path
root=Path("/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/pre-analysis");sys.path.insert(0,str(root/"Wan2.1"))
import argparse
args=argparse.ArgumentParser();args.add_argument("--case",required=True);args=args.parse_args()
registry=json.loads(Path("/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/baseline/experiments/persistence_v1/heldout_inputs.json").read_text())
case=next(c for c in registry["cases"] if c["id"]==args.case)
import torch,numpy as np
import torch.nn.functional as F
import wan
import wan.modules.model as wm
from wan.configs import WAN_CONFIGS
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
torch.set_num_threads(8)
# Exact per-sequence SDPA fallback, preserving variable key/query lengths.
def sdpa(q,k,v,q_lens=None,k_lens=None,dropout_p=0.,softmax_scale=None,q_scale=None,causal=False,window_size=(-1,-1),deterministic=False,dtype=torch.bfloat16,version=None):
 assert window_size==(-1,-1)
 b,lq,h,d=q.shape;result=torch.zeros_like(q)
 for i in range(b):
  nq=int(q_lens[i]) if q_lens is not None else lq;nk=int(k_lens[i]) if k_lens is not None else k.shape[1]
  qi=q[i,:nq].transpose(0,1).to(dtype);ki=k[i,:nk].transpose(0,1).to(dtype);vi=v[i,:nk].transpose(0,1).to(dtype)
  if q_scale is not None:qi=qi*q_scale
  y=F.scaled_dot_product_attention(qi[None],ki[None],vi[None],dropout_p=dropout_p,is_causal=causal,scale=softmax_scale)
  result[i,:nq]=y[0].transpose(0,1).to(result.dtype)
 return result
wm.flash_attention=sdpa
out=Path("/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/SceneGenerator/runs/v10_courtyard_early_inputs_r1")
out.mkdir(parents=True,exist_ok=False)
prompt=case["prompt"]
print("Loading Wan 1.3B",flush=True)
pipeline=wan.WanT2V(config=WAN_CONFIGS["t2v-1.3B"],checkpoint_dir=str(root/"weights/Wan-AI/Wan2.1-T2V-1.3B"),device_id=0)
attention_test_q=torch.randn(1,8,2,16,device="cuda",dtype=torch.bfloat16)
assert torch.isfinite(sdpa(attention_test_q,attention_test_q,attention_test_q)).all()
original=FlowUniPCMultistepScheduler.step;selected={5,10,20};records=[];counter=0;decode_seconds=0.;start=time.perf_counter()
def hooked(scheduler,model_output,timestep,sample,*args,**kwargs):
 global counter,decode_seconds
 counter+=1
 if counter in selected:
  idx=scheduler.step_index if scheduler.step_index is not None else scheduler.index_for_timestep(timestep)
  sigma=scheduler.sigmas[idx].to(sample.device)
  x0=sample-sigma*model_output
  torch.cuda.synchronize();elapsed=time.perf_counter()-start-decode_seconds
  np.save(out/f"latent_step_{counter:03d}.npy",x0[0].float().cpu().numpy())
  t=time.perf_counter()
  with torch.inference_mode():rgb=pipeline.vae.decode([x0[0]])[0]
  torch.cuda.synchronize();decode_seconds+=time.perf_counter()-t
  a=((rgb.float().clamp(-1,1)+1)/2).permute(1,2,3,0).cpu().numpy()
  np.save(out/f"rgb_step_{counter:03d}.npy",a)
  records.append({"step":counter,"nfe":counter*2,"sigma":float(sigma),"elapsed_excluding_preview_decode_seconds":elapsed})
  print("Saved",records[-1],flush=True)
 return original(scheduler,model_output,timestep,sample,*args,**kwargs)
FlowUniPCMultistepScheduler.step=hooked
# 49 frames to make lateral camera motion observable.
rgb=pipeline.generate(prompt,size=(480,480),frame_num=49,shift=5,sample_solver="unipc",sampling_steps=50,guide_scale=5.,seed=case["seed"],offload_model=True)
np.save(out/"rgb_final.npy",((rgb.float().clamp(-1,1)+1)/2).permute(1,2,3,0).cpu().numpy())
(out/"run.json").write_text(json.dumps({"model":"Wan2.1-T2V-1.3B","prompt":prompt,"seed":case["seed"],"frames":49,"size":[480,480],"steps":50,"guidance_scale":5.,"records":records,"preview_decode_seconds":decode_seconds,"attention":"PyTorch SDPA with exact variable-length masks via slicing","peak_gpu_GiB":torch.cuda.max_memory_allocated()/1024**3},indent=2))
print("WAN COMPLETE",flush=True)

old=Path("/home/tione/notebook/home/keithanwang/vlm-vla/gafu/sceneblocker/baseline/runs/persistence_unseen_inputs/courtyard_occlusion")
report={}
for kind in ["rgb","latent"]:
 aa=np.load(out/f"{kind}_step_020.npy");bb=np.load(old/f"{kind}_step_020.npy")
 report[kind]={"bitwise_equal":bool(np.array_equal(aa,bb)),"mean_abs":float(np.mean(np.abs(aa-bb))),"max_abs":float(np.max(np.abs(aa-bb)))}
(out/"replay_verification.json").write_text(json.dumps(report,indent=2));print(report,flush=True)
