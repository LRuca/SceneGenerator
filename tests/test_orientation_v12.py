import numpy as np
import unittest
from scene_baseline.orientation_v12 import recover,resolve_branch_for_smoothing,axial_delta

def test_axis_swaps_and_pi_sign_do_not_accumulate_rotation():
 t=np.linspace(0,3,49);truth=np.deg2rad(10+4*np.sin(t));obs=truth.copy();obs[[15,28,34,35]]+=np.pi/2;obs[::3]+=np.pi
 y,report=recover(t,obs,np.ones(49));assert np.max(np.abs(axial_delta(y,truth)))<np.deg2rad(4)
 assert set([15,28,34,35])<=set(report['rejected_frames']);assert np.ptp(y)<np.deg2rad(15)

def test_wrap_boundary_does_not_create_turn():
 y=np.deg2rad([85,87,89,-89,-87,-85]);out=resolve_branch_for_smoothing(y,np.ones(6));assert np.max(np.abs(np.diff(out)))<np.deg2rad(3)

def test_unresolved_branch_requires_review():
 with unittest.TestCase().assertRaises(ValueError):resolve_branch_for_smoothing(np.deg2rad([0,2,1,89]),np.ones(4))

class OrientationTests(unittest.TestCase):
 def test_axis_swap(self):test_axis_swaps_and_pi_sign_do_not_accumulate_rotation()
 def test_wrap(self):test_wrap_boundary_does_not_create_turn()
 def test_ambiguous(self):test_unresolved_branch_requires_review()

if __name__=="__main__":unittest.main()
