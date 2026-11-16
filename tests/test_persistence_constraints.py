import unittest
import numpy as np
from scene_baseline.physical_audit import box_overlap
from scene_baseline.persistence import fit_curve,evaluate_curve,DEFAULT_POLICY
from scene_baseline.geometry import box_matrix
class MotionConstraintsTest(unittest.TestCase):
 def test_wrapped_orientation_unwrap_preserves_short_turn(self):
  t=np.linspace(0,1,11);yaw=np.linspace(3,3.4,11)
  wrapped=(yaw+np.pi)%(2*np.pi)-np.pi
  c=fit_curve(t,np.unwrap(wrapped),np.ones(11),.005)
  v=evaluate_curve(c,np.linspace(0,1,41))[:,0]
  self.assertLess(np.max(np.abs(np.diff(v))),.05)
  self.assertTrue(np.allclose(v[[0,-1]],[3,3.4],atol=1e-6))
 def test_conservative_collision_detection(self):
  a=box_matrix([0,0,0],[2,1,1]);b=box_matrix([4,0,0],[2,1,1]);c=box_matrix([1,0,0],[2,1,1])
  self.assertEqual(box_overlap(a,b),0)
  self.assertGreater(box_overlap(a,c),0)
