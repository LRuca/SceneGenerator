"""Conservative axial orientation recovery for weakly observed rigid boxes.
No semantic front or turning is inferred from a partial visible surface alone.
"""
import numpy as np
from scipy.spatial.transform import Rotation

def axial_delta(a,b):
 return .5*np.angle(np.exp(2j*(np.asarray(a)-b)))

def recover(times,angles,weights,max_deviation_deg=35):
 t=np.asarray(times,float);y=np.asarray(angles,float);w=np.asarray(weights,float)
 if len(t)<2 or np.any(np.diff(t)<=0) or not np.isfinite(y).all() or np.any(w<0) or w.sum()<=0:raise ValueError('Invalid orientation evidence')
 # Weighted axial medoid: retain a supported branch before any sequential unwrap.
 cost=(np.abs(axial_delta(y[:,None],y[None,:]))*w[:,None]).sum(0)
 anchor=float(y[np.argmin(cost)]);d=axial_delta(y,anchor)
 keep=(np.abs(d)<=np.deg2rad(max_deviation_deg))&(w>=max(float(np.max(w))*.05,1e-12))
 if keep.sum()<4:raise ValueError('Insufficient consistent orientation evidence; agent review required')
 # Low-capacity straight-axis hypothesis. Rejecting measurements is not proof they are wrong.
 # No turns are claimed in this fallback; retain measured variation within the dominant mode.
 from .persistence import fit_curve,evaluate_curve
 curve=fit_curve(t[keep],(anchor+d[keep]),w[keep],.01)
 out=evaluate_curve(curve,np.clip(t,t[keep][0],t[keep][-1]))[:,0]
 return out,dict(anchor_rad=anchor,accepted_frames=np.flatnonzero(keep).tolist(),rejected_frames=np.flatnonzero(~keep).tolist(),input_yaw_rad=y.tolist(),weights=w.tolist(),output_yaw_rad=out.tolist(),hypothesis='dominant axial mode; no demonstrated turn; clipped endpoint orientation',semantic_front_resolved=False,limitation='coherent real turns beyond35 degrees require another hypothesis or verified part correspondences; cannot identify an axis from depth support alone')

def resolve_branch_for_smoothing(angles,weights):
 """Smooth an axis in projective space, not by chaining noisy unwrap decisions."""
 y=np.asarray(angles,float);w=np.asarray(weights,float)
 anchor=y[np.argmin((np.abs(axial_delta(y[:,None],y[None,:]))*w[:,None]).sum(0))]
 d=axial_delta(y,anchor)
 if np.max(np.abs(d))>np.deg2rad(60):
  raise ValueError('Ambiguous axial branch: resolve orientation evidence before smoothing')
 return anchor+d
