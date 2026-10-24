"""Pinned pre-release UI enrichment rejects mismatched evidence."""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path.cwd()
spec=importlib.util.spec_from_file_location("case_details_service",ROOT/"webapp/backend/app/components/c4_service.py")
service=importlib.util.module_from_spec(spec);spec.loader.exec_module(service)

class DetailTests(unittest.TestCase):
 def fixture(self):
  upstream={"upstream":{"c1_evidence":{},"c2_diagnosis":{},"c3_repair_package":{}},"source_declarations":{"c1":"mock_oracle"},"privacy_interpretation":"simulation only"}
  frozen={"upstream_simulation_sha256":"up","pre_release_evidence_sha256":"evidence", "scenario":{"scenario_id":"test"},
   "dependencies":{r"C:\repo\components\c4\scripts\_scenario_meta.py":"helper",r"G:\data\test\scoped_candidate_validation-v1\metadata.json":"candidate"}}
  records={"freeze":(frozen,"freeze"),"upstream_simulation":(upstream,"up"),"evidence":({"report":{"comparisons":{"global":{"rows":10}}}},"evidence"),"metadata":({"champion_version":"champion-id","artifact_version":"candidate-id"},"candidate")}
  common=SimpleNamespace(resolve_data_root=lambda _:Path("/test"),checked_json=lambda path,stem="metadata":records[stem],checked_bytes=lambda *a:b"pinned")
  features=SimpleNamespace(feature_row=lambda *a:{"x":1.})
  return records,common,features
 def run_details(self,common,features):
  with patch.object(service.importlib,"import_module",side_effect=lambda name:common if name=="_common" else features):
   return service._case_details({"evidence_id":"test","features":{"x":1.}})
 def test_pinned_records_return_actual_identifiers(self):
  records,common,features=self.fixture();out=self.run_details(common,features)
  self.assertEqual(out["candidate_model_id"],"candidate-id")
  self.assertEqual(out["candidate_evaluation"]["global"]["rows"],10)
  self.assertTrue(out["displayed_evidence_reconstructed"])
 def test_upstream_mismatch_rejected(self):
  records,common,features=self.fixture();records["upstream_simulation"]=(records["upstream_simulation"][0],"wrong")
  with self.assertRaises(service.C4Unavailable):self.run_details(common,features)
 def test_pre_release_mismatch_rejected(self):
  records,common,features=self.fixture();records["evidence"]=(records["evidence"][0],"wrong")
  with self.assertRaises(service.C4Unavailable):self.run_details(common,features)
 def test_reconstruction_mismatch_rejected(self):
  records,common,features=self.fixture();features.feature_row=lambda *a:{"x":2.}
  with self.assertRaises(service.C4Unavailable):self.run_details(common,features)
 def test_candidate_link_mismatch_rejected(self):
  records,common,features=self.fixture();records["metadata"]=(records["metadata"][0],"wrong")
  with self.assertRaises(service.C4Unavailable):self.run_details(common,features)
 def test_unbound_reconstructor_rejected(self):
  records,common,features=self.fixture();records["freeze"][0]["dependencies"]={}
  with self.assertRaises(service.C4Unavailable):self.run_details(common,features)
