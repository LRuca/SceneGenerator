import copy,unittest
import numpy as np
from scene_baseline.modeling_policy_v4 import validate,execution_plan,entity_budget
from scene_baseline.cage_v4 import make_state,local_vertices,torch_vertices,validate as validate_cage
from scene_baseline.geometry import scene_parts
class TestPolicy(unittest.TestCase):
 def setUp(self):
  self.scene={"entities":[{"id":"actor","motion":"rigid","parts":[{}]},{"id":"wall","motion":"static","parts":[{}]},{"id":"decor","motion":"static","parts":[{}]}]}
  self.policy={"schema":"scene_baseline.modeling_policy.v4","actor":"test","entities":[]}
  for eid,role,duties,rep in [("actor","key_actor",[],"primitive_assembly"),("wall","incidental",["occlusion"],"thin_surface"),("decor","incidental",[],"description_only")]:
   self.policy["entities"].append(dict(id=eid,narrative_role=role,geometric_duties=duties,representation=rep,confidence=.8,evidence=["indexed source frame"],part_budget=1 if rep!="description_only" else 0,motion="rigid" if eid=="actor" else "static",identity_status="reviewed",boundary_status="reviewed"))
 def test_role_and_duty_independent(self):
  validate(self.policy,self.scene)
  for e,p in zip(self.scene["entities"],self.policy["entities"]):e["modeling_policy"]=p
  plan=execution_plan(self.scene,900);jobs={j["entity_id"]:j for j in plan["jobs"]}
  self.assertEqual(sum(j["steps"] for j in jobs.values()),900)
  self.assertGreater(jobs["actor"]["steps"],jobs["wall"]["steps"])
  self.assertTrue(jobs["wall"]["retain_geometry"]);self.assertFalse(jobs["decor"]["retain_geometry"])
  self.scene["modeling_execution_plan"]=plan
  self.assertEqual(entity_budget(self.scene,"actor",1000),600)
  with self.assertRaises(ValueError):entity_budget(self.scene,"decor",100)
 def test_no_description_for_structural_or_key_entity(self):
  for i in [0,1]:
   p=copy.deepcopy(self.policy);p["entities"][i]["representation"]="description_only"
   with self.assertRaises(ValueError):validate(p,self.scene)
 def test_revision_gate_and_identity(self):
  p=copy.deepcopy(self.policy);p["entities"][0].update(motion="deformable",representation="deformable_cage")
  validate(p,self.scene)
  for e,x in zip(self.scene["entities"],p["entities"]):e["modeling_policy"]=x
  self.assertEqual(execution_plan(self.scene)["requires_revision"],["actor"])
  p["entities"][0]["id"]="other"
  with self.assertRaises(ValueError):validate(p,self.scene)
 def test_description_excluded_without_deleting_identity(self):
  e={"id":"decor","modeling_policy":{"representation":"description_only"}}
  self.assertEqual(scene_parts({"entities":[e]},0),[])
  self.assertEqual(e["id"],"decor")
class TestCage(unittest.TestCase):
 def test_nonzero_radial_numpy_torch_match(self):
  import torch
  s=make_state([.6,1.8,.4],np.random.default_rng(0).normal(0,.02,(3,5,2)),["observed"]*3)
  s["radial_log"]=np.random.default_rng(1).normal(0,.03,(3,5,2)).tolist();validate_cage(s,3)
  actual=torch_vertices(torch.tensor(s["shared_size"],dtype=torch.float64),torch.tensor(s["bend_xz"],dtype=torch.float64),radial_log=torch.tensor(s["radial_log"],dtype=torch.float64)).numpy()
  self.assertLess(np.max(abs(actual-np.stack([local_vertices(s,t) for t in range(3)]))),1e-12)
 def test_smooth_axis_geometry_agrees(self):
  import torch
  s=make_state([.6,1.8,.4],np.random.default_rng(3).normal(0,.02,(3,5,2)),["observed"]*3)
  s["axis_mode"]="smooth_v5"
  actual=torch_vertices(torch.tensor(s["shared_size"],dtype=torch.float64),torch.tensor(s["bend_xz"],dtype=torch.float64),smooth=True).numpy()
  expected=np.stack([local_vertices(s,t) for t in range(3)])
  self.assertEqual(actual.shape,(3,204,3))
  self.assertLess(np.max(abs(actual-expected)),1e-12)
 def test_rigid_modes_not_deformation(self):
  s=make_state([.6,1.8,.4],np.zeros((3,5,2)),["prior"]*3)
  s["bend_xz"][0][0][0]=.1
  with self.assertRaises(ValueError):validate_cage(s,3)
if __name__=="__main__":unittest.main()

class TestHumanHinge(unittest.TestCase):
 def test_two_joint_and_head_shared_geometry(self):
  import torch
  from scene_baseline.human_hinge_v6 import vertices,validate,parts,SCHEMA,SECTIONS,CIRCUMFERENCE
  angles=torch.tensor([[0.,0.],[.1,-1.]],dtype=torch.float64)
  body,head,headpoints,axis=vertices(torch.tensor(1.8),torch.tensor(.65),torch.tensor(.45),angles)
  self.assertEqual(tuple(body.shape),(2,SECTIONS*CIRCUMFERENCE,3))
  self.assertTrue(np.allclose(axis[:,0].numpy(),0))
  self.assertTrue(np.allclose(np.linalg.norm((head-axis[:,-1]).numpy(),axis=-1),1.8*.08))
  state=dict(schema=SCHEMA,shared_size=[.65,1.8,.45],angles=angles.tolist(),state_source=["test"]*2)
  validate(state,2)
  e=dict(id="person",deformation=state,entity_to_world=np.tile(np.eye(4),(2,1,1)).tolist())
  pp=parts(e,1)
  self.assertEqual([x["component"] for x in pp],["body","head"])
  self.assertTrue(np.allclose(np.asarray(pp[0]["vertices"]),body[1].numpy(),atol=1e-6))
  state["angles"][0][0]=1.
  with self.assertRaises(ValueError):validate(state,2)
 def test_projected_triangle_union_does_not_cancel(self):
  from scene_baseline.envelope_v6 import projected_mask
  p=dict(kind="cage",entity_id="x",vertices=[[-1,-1,2],[1,-1,2],[1,1,2],[-1,1,2]],faces=[[0,1,2],[0,2,3],[0,1,2]],canonical_to_world=np.eye(4).tolist())
  K=np.array([[10,0,10],[0,10,10],[0,0,1.]])
  m=projected_mask([p],K,np.eye(4),21,21,"x")
  self.assertTrue(m[10,10]);self.assertGreater(m.sum(),100)
