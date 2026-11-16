import unittest,tempfile
from pathlib import Path
import numpy as np
from scene_baseline.storage import BASE,inside,reference,verify,write_json
from scene_baseline.tools import invoke
from scene_baseline.contracts import validate_geometry,validate_masks
from scene_baseline.geometry import primitive_depth,box_matrix,agreement

class ContractsToolsTest(unittest.TestCase):
    def setUp(self):
        (BASE/"runs/tests").mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=BASE/"runs/tests");self.p=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def test_outside_write_blocked(self):
        with self.assertRaises(ValueError):inside("/tmp/not-authorized")
        link=self.p/"escape";link.symlink_to("/tmp")
        with self.assertRaises(ValueError):inside(link/"x")
    def test_hash_detects_changed_input(self):
        p=self.p/"a";p.write_text("a");ref=reference(p);p.write_text("b")
        with self.assertRaises(ValueError):verify(ref)
    def test_tool_immutable_and_parent_binding(self):
        p=self.p/"rgb.npy";np.save(p,np.random.default_rng(7).random((2,16,16,3)).astype("float32"))
        original=reference(p);request={"tool":"rgb.bilateral","modality":"rgb","input":original,"actor":"test","reason":"component test"}
        write_json(self.p/"request.json",request)
        r=invoke(self.p/"request.json",self.p/"out")
        self.assertEqual(original,reference(p));self.assertEqual(r["parent"],original)
        self.assertEqual(r["coordinate_effect"],"identity")
        with self.assertRaises(FileExistsError):invoke(self.p/"request.json",self.p/"out")
    def test_invalid_parameters_rejected(self):
        p=self.p/"rgb.npy";np.save(p,np.zeros((2,8,8,3),dtype="float32"))
        req={"tool":"rgb.unsharp","modality":"rgb","input":reference(p),"parameters":{"amount":-1},"actor":"test","reason":"test"}
        write_json(self.p/"req.json",req)
        with self.assertRaises(ValueError):invoke(self.p/"req.json",self.p/"out")
        self.assertFalse((self.p/"out").exists())
    def test_camera_z_not_ray_length(self):
        K=np.array([[20.,0,16],[0,20,16],[0,0,1.]])
        d=primitive_depth("box",box_matrix([0,0,3],[2,2,1]),K,np.eye(4),32,32)
        self.assertAlmostEqual(d[16,16],2.5);self.assertAlmostEqual(d[16,20],2.5)
    def test_missing_prediction_penalized(self):
        r=agreement(np.full((4,4),np.inf),np.ones((4,4)))
        self.assertEqual(r["coverage"],0);self.assertEqual(r["capped_relative_error_with_misses"],1)
    def test_single_view_contract_and_mismatched_masks(self):
        a={"rgb":np.zeros((2,4,4,3)),"depth":np.ones((2,4,4)),"confidence":np.ones((2,4,4)),
            "K":np.tile(np.eye(3),(2,1,1)),"c2w":np.tile(np.eye(4),(2,1,1)),
            "times_s":np.array([0.,1.]),"original_to_model":np.tile(np.eye(3),(2,1,1))}
        self.assertEqual(validate_geometry(a),(2,4,4))
        with self.assertRaises(ValueError):validate_masks(np.zeros((1,1,4,4),bool),["car"],a)
        a["c2w"][1,0,0]=2
        with self.assertRaises(ValueError):validate_geometry(a)


    def test_report_tool_and_depth_branch(self):
        depth=self.p/"depth.npy";np.save(depth,np.ones((2,8,8),dtype="float32")*3)
        for i,tool in enumerate(["depth.inspect_quality","depth.spatial_log_median"]):
            request={"tool":tool,"modality":"depth","input":reference(depth),"actor":"test","reason":"depth branch"}
            req=self.p/f"req{i}.json";write_json(req,request)
            r=invoke(req,self.p/f"out{i}")
            self.assertEqual(r["parent"],reference(depth))
            self.assertTrue(Path(r["output"]["path"]).is_file())
        filtered=np.load(self.p/"out1/array.npy")
        self.assertTrue(np.allclose(filtered,3))

    def test_unknown_tool_not_executed(self):
        p=self.p/"bad.json";write_json(p,{"tool":"shell.execute","command":"anything"})
        with self.assertRaises(KeyError):invoke(p,self.p/"out")
        self.assertFalse((self.p/"out").exists())


    def test_pipeline_unknown_stage_recorded(self):
        from scene_baseline.pipeline import run
        config=self.p/"pipeline.json"
        write_json(config,{"schema":"scene_baseline.pipeline.v1","tasks":[{"id":"unsafe","kind":"shell.execute","arguments":{}}]})
        with self.assertRaises(ValueError):run(config,self.p/"pipeline")
        import json
        status=json.loads((self.p/"pipeline/unsafe.status.json").read_text())
        self.assertEqual(status["status"],"failed")

    def test_pixel_resize_roundtrip(self):
        scale=518/480
        A=np.array([[scale,0,(scale-1)/2],[0,scale,(scale-1)/2],[0,0,1.]])
        pixel=np.array([123.,246.,1.])
        self.assertTrue(np.allclose(np.linalg.inv(A)@(A@pixel),pixel))
