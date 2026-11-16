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
 def test_rigid_modes_not_deformation(self):
  s=make_state([.6,1.8,.4],np.zeros((3,5,2)),["prior"]*3)
  s["bend_xz"][0][0][0]=.1
  with self.assertRaises(ValueError):validate_cage(s,3)
if __name__=="__main__":unittest.main()
