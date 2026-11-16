import unittest,tempfile,copy
from pathlib import Path
import numpy as np
import torch
from scene_baseline.storage import BASE,read_json,write_json,reference
from scene_baseline.geometry import primitive_depth,rays,box_matrix
from scene_baseline.fit_scene import _intersection,fit
from scene_baseline.scene import initialize,load_scene,validate_scene
from scene_baseline.fixtures import create
from scene_baseline.delivery import render
from scene_baseline.revision import apply

class ScenePipelineTest(unittest.TestCase):
    def setUp(self):
        (BASE/"runs/tests").mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=BASE/"runs/tests");self.p=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_torch_intersections_match_reference(self):
        K=np.array([[35.,0,16],[0,35.,16],[0,0,1.]])
        matrix=box_matrix([0,0,3],[1.2,1.4,.8])
        origin,direction=rays(K,np.eye(4),32,32);inv=np.linalg.inv(matrix)
        o=np.broadcast_to(inv[:3,:3]@origin+inv[:3,3],direction.shape)
        d=direction@inv[:3,:3].T
        for kind in ["box","sphere","cylinder","cone","plane"]:
            actual=_intersection(torch.tensor(o[None],dtype=torch.float32),torch.tensor(d[None],dtype=torch.float32),kind)[0].numpy().reshape(32,32)
            expected=primitive_depth(kind,matrix,K,np.eye(4),32,32)
            hit=np.isfinite(expected)
            self.assertTrue(np.array_equal(actual<1e5,hit),kind)
            self.assertTrue(np.allclose(actual[hit],expected[hit],atol=2e-4),kind)
    def test_scene_initialization_and_revision(self):
        source=create(self.p/"fixture",n=3,size=32)
        doc=initialize(source/"geometry_manifest.json",source/"hypothesis.json",source/"masks_manifest.json",self.p/"init")
        self.assertEqual(len(doc["entities"]),3)
        request={"schema":"scene_baseline.revision.v1","parent_scene":reference(self.p/"init/scene.json"),"actor":"test","reason":"topology refinement",
            "actions":[{"type":"replace_parts","entity_id":"vehicle","parts":doc["entities"][-1]["parts"]}]}
        write_json(self.p/"rev.json",request)
        apply(self.p/"init/scene.json",self.p/"rev.json",self.p/"revised")
        self.assertEqual(read_json(self.p/"init/scene.json"),doc)
        request["parent_scene"]["sha256"]="bad";write_json(self.p/"bad.json",request)
        with self.assertRaises(ValueError):apply(self.p/"init/scene.json",self.p/"bad.json",self.p/"bad")
    def test_fit_and_render_shared_scene(self):
        source=create(self.p/"fixture",n=3,size=32)
        report=fit(source/"initial_scene.json",self.p/"fit",steps=80,resolution=24,device="cpu",lr=.012)
        self.assertLess(report["best_loss"],report["history"][0]["loss"]*.8)
        output=render(self.p/"fit/scene.json",self.p/"delivery",width=48)
        self.assertGreater(output["mean_coverage"],.85)
        self.assertLess(output["mean_capped_relative_error_with_misses"],.12)
        self.assertTrue((self.p/"delivery/comparison.mp4").is_file())
        scene,_,_=load_scene(self.p/"fit/scene.json")
        vehicle=next(e for e in scene["entities"] if e["id"]=="vehicle")
        self.assertEqual(len(vehicle["entity_to_world"]),3)
        self.assertEqual(len(vehicle["parts"]),1)


    def test_depth_tool_adoption_is_explicit(self):
        from scene_baseline.tools import invoke
        from scene_baseline.observations import derive_depth
        from scene_baseline.contracts import load_bundle
        source=create(self.p/"fixture",n=2,size=32)
        manifest=source/"geometry_manifest.json";_,a=load_bundle(manifest)
        np.save(source/"depth.npy",a["depth"])
        request={"tool":"depth.spatial_log_median","modality":"depth","input":reference(source/"depth.npy"),
            "geometry_manifest":reference(manifest),"actor":"test","reason":"branch test","usage":"agent_preview"}
        write_json(self.p/"preview_req.json",request)
        invoke(self.p/"preview_req.json",self.p/"preview_tool")
        with self.assertRaises(ValueError):derive_depth(manifest,self.p/"preview_tool/result.json",self.p/"forbidden")
        request["usage"]="geometry_candidate";write_json(self.p/"candidate_req.json",request)
        invoke(self.p/"candidate_req.json",self.p/"candidate_tool")
        derive_depth(manifest,self.p/"candidate_tool/result.json",self.p/"derived")
        _,b=load_bundle(self.p/"derived/manifest.json")
        self.assertTrue(np.array_equal(a["c2w"],b["c2w"]))
        self.assertTrue(np.array_equal(a["K"],b["K"]))
        self.assertTrue(np.all(b["confidence"]==1))
        self.assertTrue(np.array_equal(a["confidence"],b["raw_confidence"]))

    def test_hidden_entity_and_wrong_bundle(self):
        from scene_baseline.geometry import scene_parts
        from scene_baseline.scene import load_observations
        source=create(self.p/"fixture",n=2,size=32)
        scene=read_json(source/"truth_scene.json")
        scene["entities"][-1]["renderable"][0]=False
        self.assertEqual(len(scene_parts(scene,0)),2)
        other=create(self.p/"other",n=2,size=32)
        with self.assertRaises(ValueError):
            load_observations(other/"geometry_manifest.json",source/"hypothesis.json",source/"masks_manifest.json")
