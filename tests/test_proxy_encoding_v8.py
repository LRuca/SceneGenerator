import copy
import unittest
import numpy as np
from scene_baseline.proxy_encoding_v8 import encode,camera_path,binding_prompt

class EncodingTests(unittest.TestCase):
    def fixture(self):
        scene={'entities':[{'id':'moving','label':'forklift','motion':'rigid','parts':[{'shape':'unchanged'}]}, {'id':'parked','label':'pickup','motion':'static','parts':[{'shape':'unchanged'}]}, {'id':'wall','label':'wall','motion':'static','parts':[{'shape':'unchanged'}]}]}
        decisions={'actor':'test_agent','entities':[{'id':eid,'role':role,'appearance_image':i,'evidence':'test','geometric_duties':['occlusion']} for eid,role,i in [('moving','key_actor',1),('parked','key_prop',2),('wall','structural',3)]]}
        return scene,decisions
    def test_static_key_prop_distinct_without_geometry_change(self):
        scene,decisions=self.fixture();before=copy.deepcopy(scene);out=encode(scene,decisions)
        self.assertEqual(scene,before)
        colors=[e['proxy_encoding']['rgb'] for e in out['entities']]
        self.assertNotEqual(colors[0],colors[1]);self.assertEqual(colors[2],[185]*3)
        for a,b in zip(out['entities'],before['entities']):
            self.assertEqual(a['parts'],b['parts']);self.assertEqual(a['motion'],b['motion'])
        prompt=binding_prompt(out['proxy_encoding']);self.assertIn('BLUE',prompt);self.assertIn('[Image 2]',prompt);self.assertIn('BLUE / parked has one shared world pose',prompt)
    def test_incomplete_or_unbound_identity_rejected(self):
        s,d=self.fixture();d['entities'].pop()
        with self.assertRaises(ValueError):encode(s,d)
        s,d=self.fixture();d['entities'][0]['appearance_image']=None
        with self.assertRaises(ValueError):encode(s,d)
    def test_translation_preserves_rotation_first_pose_and_ground_height(self):
        base=np.tile(np.eye(4),(9,1,1));base[:,2,3]=np.linspace(0,1,9)
        before=base.copy();cams,right=camera_path(base,[0,1,0],-.2)
        np.testing.assert_array_equal(base,before);np.testing.assert_array_equal(cams[:,:3,:3],base[:,:3,:3])
        np.testing.assert_array_equal(cams[0],base[0]);np.testing.assert_allclose(cams[-1,:3,3]-base[-1,:3,3],[-.2,0,0])
        self.assertTrue(np.all(np.diff(cams[:,0,3])<=0));self.assertEqual(right[1],0)

    def test_appearance_swap_does_not_leak_source_class_into_prompt_binding(self):
        scene,d=self.fixture()
        d['entities'][0].update(appearance_label='green coupe',prompt_id='I1')
        out=encode(scene,d)
        self.assertEqual(out['entities'][0]['label'],'forklift')
        prompt=binding_prompt(out['proxy_encoding'])
        self.assertIn('green coupe (instance I1)',prompt)
        self.assertNotIn('forklift',prompt)
