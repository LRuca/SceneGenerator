import unittest,tempfile,copy
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scene_baseline.persistence import fit_curve,evaluate_curve,apply,DEFAULT_POLICY,sample_entity
from scene_baseline.storage import BASE,write_json,read_json
from scene_baseline.fixtures import create
from scene_baseline.geometry import scene_parts
from scene_baseline.scene import validate_scene

class PersistenceTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=BASE/"runs/tests");self.p=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def test_constant_velocity_missing_boundaries_and_interior(self):
  t=np.array([.3,.45,.7,1.1,1.35]);y=np.c_[t*2+1,t*0+2,t*-.5]
  c=fit_curve(t,y,np.ones(len(t)),.0025)
  q=np.array([0,.3,.6,1.35,1.8]);pred=evaluate_curve(c,q)
  self.assertTrue(np.allclose(pred,np.c_[q*2+1,q*0+2,q*-.5],atol=1e-6))
  for edge in [t[0],t[-1]]:
   for order in [0,1,2]:
    v=evaluate_curve(c,np.array([edge-1e-6,edge+1e-6]),order)
    self.assertLess(np.linalg.norm(v[0]-v[1]),1e-4)
 def test_noise_reduction_and_unseen_contiguous_interval(self):
  rng=np.random.default_rng(541);t=np.linspace(0,3,61)
  truth=np.c_[.3*t,np.sin(t)*.2,t*0]
  noisy=truth+rng.normal(0,.025,truth.shape);train=(t<1.2)|(t>1.8)
  c=fit_curve(t[train],noisy[train],np.ones(train.sum()),.0025)
  pred=evaluate_curve(c,t)
  self.assertLess(np.mean((pred[~train]-truth[~train])**2),.002)
  before=np.diff(noisy,n=2,axis=0);after=np.diff(pred,n=2,axis=0)
  self.assertLess(np.mean(after**2),np.mean(before**2)*.1)
 def test_mask_absence_does_not_delete_and_explicit_lifespan(self):
  f=create(self.p/"f",n=9,size=32)
  scene=read_json(f/"initial_scene.json");e=scene["entities"][-1]
  self.assertEqual(e["motion"],"rigid")
  e["lifespan_events"]={"birth_s":.1,"death_s":.8}
  # Test with ordinary fixture times first: custom event outside interval is valid.
  write_json(self.p/"source.json",scene);write_json(self.p/"policy.json",DEFAULT_POLICY)
  new=apply(self.p/"source.json",self.p/"policy.json",self.p/"run")
  e=new["entities"][-1]
  self.assertEqual(e["renderable"],e["exists"])
  self.assertTrue(any(not x for x in e["exists"]))
 def test_so3_and_ground_contact_and_uncertainty(self):
  f=create(self.p/"f",n=20,size=32);scene=read_json(f/"initial_scene.json")
  # Remove evidence at beginning, middle, end, preserving actual immutable source linkage.
  from scene_baseline.storage import reference
  md=read_json(f/"masks_manifest.json")
  z=np.load(md["data"]["path"]);m=z["masks"].copy();ids=z["ids"].copy()
  m[-1,:3]=False;m[-1,8:11]=False;m[-1,-3:]=False
  np.savez_compressed(self.p/"m.npz",masks=m,ids=ids)
  md["data"]=reference(self.p/"m.npz");write_json(self.p/"md.json",md)
  scene["masks_manifest"]=reference(self.p/"md.json")
  write_json(self.p/"s.json",scene);write_json(self.p/"policy.json",DEFAULT_POLICY)
  new=apply(self.p/"s.json",self.p/"policy.json",self.p/"new")
  e=new["entities"][-1];self.assertTrue(all(e["exists"]));self.assertFalse(e["observation_available"][0])
  self.assertEqual(e["state_source"][0],"inferred_extrapolation")
  self.assertEqual(e["state_source"][9],"inferred_interpolation")
  self.assertEqual(len(scene_parts(new,0)),len(scene_parts(new,5)))
  t=np.linspace(0,19/16,201);pose=sample_entity(e,t);R=pose[:,:3,:3]
  self.assertTrue(np.allclose(R.transpose(0,2,1)@R,np.eye(3),atol=1e-6))
  uncertainty=e["uncertainty_proxy"]["position_radius"];self.assertGreater(uncertainty[0],uncertainty[3])
  validate_scene(new,20)
 def test_reject_wrong_timeline_and_irregular_curve(self):
  with self.assertRaises(ValueError):fit_curve([0,.1,.1],np.ones((3,1)),np.ones(3),.1)
  f=create(self.p/"f",n=5,size=32);s=read_json(f/"initial_scene.json")
  s["entities"][-1]["state_source"]=["x"]*5
  with self.assertRaises((ValueError,KeyError)):validate_scene(s,5)
