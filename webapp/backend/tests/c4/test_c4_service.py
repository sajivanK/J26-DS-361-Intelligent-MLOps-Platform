"""Dependency-free tests for the new C4 page service."""
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path.cwd()
spec=importlib.util.spec_from_file_location("c4_service_test",ROOT/"webapp/backend/app/components/c4_service.py")
service=importlib.util.module_from_spec(spec);spec.loader.exec_module(service)

class ServiceTests(unittest.TestCase):
 def test_catalog_explicitly_excludes_live(self):
  self.assertFalse(service.catalog()["live_upstream_integration"])
  self.assertFalse(service.catalog()["production_release_authorized"])
 def test_research_counts(self):
  d=service.recorded_results();self.assertEqual(d["counts"]["admitted"]+d["counts"]["blocked"],592)
  row=next(r for r in d["governance"] if r["comparison"]=="PROPOSED")
  self.assertEqual(int(row["count_review"]),56)
  self.assertEqual(row["equal_group_conditional_unsafe_rate"],"")
 def test_group_coverage(self):
  p=service.recorded_results()["intervals"]["PROPOSED"]
  self.assertEqual(p["group_covered_count"],12);self.assertEqual(p["nonempty_groups"],14)
 def test_unknown_case_before_runtime(self):
  with patch.object(service,"_runtime",side_effect=AssertionError("must not load")):
   with self.assertRaises(KeyError):service.evaluate("unknown")
 def test_bad_method_before_runtime(self):
  with self.assertRaises(ValueError):service.evaluate("real-outer-case","unknown")
 def test_actual_controller_invoked(self):
  class Controller:
   def decide(self,features,gates,**kw):
    self.received=(features,gates,kw)
    return {"decision":"HUMAN_REVIEW","production_release_authorized":False}
  from types import SimpleNamespace
  c=Controller();c.calibrations={"lightgbm":SimpleNamespace(to_dict=lambda:{"adjustment":.006})};payload={"features":{"x":1},"hard_gates":{"gate":True},"evidence_id":"test"}
  with patch.object(service,"_runtime",return_value=(c,{},"hash")),patch.object(service,"_demo",return_value=(payload,"inputhash")),patch.object(service,"_case_details",return_value={"candidate_evaluation":{}}):
   out=service.evaluate("real-outer-case")
  self.assertTrue(out["inference_executed"])
  self.assertFalse(out["candidate_retrained"])
  self.assertEqual(c.received[2]["evidence_id"],"test")
  self.assertEqual(out["result"]["decision"],"HUMAN_REVIEW")
 def test_sources_compile(self):
  for p in (ROOT/"webapp/backend").rglob("*.py"):compile(p.read_text(),str(p),"exec")
if __name__=="__main__":unittest.main()
