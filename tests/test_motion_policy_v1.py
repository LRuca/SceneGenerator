import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scene_baseline.motion_policy_v1 import validate, resolve_scene, require_locked, apply, fit_locked
from scene_baseline.storage import BASE, write_json, read_json, reference
from scene_baseline.fixtures import create

class MotionPolicyTests(unittest.TestCase):
    def scene(self):
        pose = np.repeat(np.eye(4)[None], 3, axis=0)
        return {'schema':'scene_baseline.scene.v1', 'entities':[
            {'id':'car', 'motion':'rigid', 'entity_to_world':pose.tolist(),
             'parts':[{'kind':'box','canonical_to_entity':np.eye(4).tolist()}],
             'renderable':[True]*3, 'observed':[True]*3, 'continuous_motion':{'obsolete':True}}]}
    def decisions(self, motion='static'):
        return {'actor':'reviewing_agent','entities':[{'id':'car','motion':motion,'evidence':'same-stage sequence review'}]}
    def test_missing_duplicate_and_empty_actor_rejected(self):
        s=self.scene();d=self.decisions();d['entities']=[]
        with self.assertRaises(ValueError):validate(d,s)
        d=self.decisions();d['entities']*=2
        with self.assertRaises(ValueError):validate(d,s)
        d=self.decisions();d['actor']=''
        with self.assertRaises(ValueError):validate(d,s)
    def test_unknown_cannot_silently_become_dynamic(self):
        with self.assertRaises(ValueError):resolve_scene(self.scene(),self.decisions('unknown'),3)
    def test_static_shared_pose_and_stale_curve_removed(self):
        s=self.scene();before=copy.deepcopy(s);out=resolve_scene(s,self.decisions(),3)
        self.assertEqual(s,before);self.assertEqual(len(out['entities'][0]['entity_to_world']),1)
        self.assertNotIn('continuous_motion',out['entities'][0]);require_locked(out)
    def test_static_to_dynamic_expands_timeline(self):
        s=resolve_scene(self.scene(),self.decisions(),3)
        out=resolve_scene(s,self.decisions('rigid_dynamic'),3)
        self.assertEqual(len(out['entities'][0]['entity_to_world']),3)
        require_locked(out)
    def test_semantic_mutation_and_unreviewed_scene_rejected(self):
        with self.assertRaises(ValueError):require_locked(self.scene())
        out=resolve_scene(self.scene(),self.decisions(),3);out['entities'][0]['motion']='rigid'
        with self.assertRaises(ValueError):require_locked(out)
    def test_deformation_requires_topology_revision(self):
        with self.assertRaises(ValueError):resolve_scene(self.scene(),self.decisions('deformable'),3)
    def test_wrong_source_and_unknown_rejected_without_output(self):
        (BASE/'runs/tests').mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=BASE/'runs/tests') as tmp:
            p=Path(tmp);create(p/'fixture',n=3,size=24)
            source=p/'fixture/initial_scene.json';scene=read_json(source)
            rows=[{'id':e['id'],'motion':'static','evidence':'fixture'} for e in scene['entities']]
            dec={'actor':'agent','source_scene':{'path':str(source),'sha256':'wrong'},'entities':rows}
            write_json(p/'wrong.json',dec)
            with self.assertRaises(ValueError):apply(source,p/'wrong.json',p/'wrong_out')
            self.assertFalse((p/'wrong_out').exists())
            dec['source_scene']=reference(source);dec['entities'][0]['motion']='unknown'
            write_json(p/'unknown.json',dec)
            with self.assertRaises(ValueError):apply(source,p/'unknown.json',p/'unknown_out')
            self.assertFalse((p/'unknown_out').exists())

    def test_dispatcher_apply_and_guarded_fit(self):
        from scene_baseline.pipeline import _stage
        (BASE/'runs/tests').mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=BASE/'runs/tests') as tmp:
            p=Path(tmp);create(p/'fixture',n=3,size=24)
            source=p/'fixture/initial_scene.json';scene=read_json(source)
            decisions={'source_scene':reference(source),'actor':'test_agent','entities':[{'id':e['id'],'motion':'static' if e['motion']=='static' else 'rigid_dynamic','evidence':'controlled fixture'} for e in scene['entities']]}
            write_json(p/'decisions.json',decisions)
            output=_stage('agent.motion_decision',{'scene':str(source),'decisions':str(p/'decisions.json')},p/'locked')
            with self.assertRaises(ValueError):fit_locked(source,p/'bad',steps=1,resolution=8)
            _stage('scene.fit_agent',{'scene':output['scene'],'steps':2,'resolution':8},p/'fit')
            require_locked(read_json(p/'fit/scene.json'))

if __name__=='__main__':unittest.main()
