"""C4 router tests without importing or loading C1's component registry."""
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
try:
 from fastapi import FastAPI
 from fastapi.testclient import TestClient
except ImportError:
 FastAPI=None

@unittest.skipUnless(FastAPI,"Install FastAPI and httpx in the platform test environment")
class RouterTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  root=Path.cwd()/"webapp/backend/app"
  for name,path in [("c4_test_app",root),("c4_test_app.components",root/"components")]:
   module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
  cls.route=importlib.import_module("c4_test_app.routers.c4")
  app=FastAPI();app.include_router(cls.route.router);cls.client=TestClient(app)
 def test_info(self):
  r=self.client.get("/api/c4/info");self.assertEqual(r.status_code,200)
  self.assertFalse(r.json()["live_upstream_integration"])
 def test_results(self):
  r=self.client.get("/api/c4/research-results");self.assertEqual(r.status_code,200)
  self.assertEqual(r.json()["counts"]["admitted"],56)
 def test_unknown_case(self):
  r=self.client.post("/api/c4/release/evaluate",json={"case_id":"unknown"})
  self.assertEqual(r.status_code,404)
 def test_invalid_enum(self):
  r=self.client.post("/api/c4/release/evaluate",json={"method":"invalid"})
  self.assertEqual(r.status_code,422)
 def test_extra_target_rejected(self):
  r=self.client.post("/api/c4/release/evaluate",json={"future_target":1})
  self.assertEqual(r.status_code,422)
 def test_artifact_unavailable(self):
  with patch.object(self.route.service,"evaluate",side_effect=self.route.service.C4Unavailable("Artifacts unavailable")):
   r=self.client.post("/api/c4/release/evaluate",json={})
  self.assertEqual(r.status_code,503)
 def test_backend_failure_is_not_fake_decision(self):
  with patch.object(self.route.service,"evaluate",side_effect=RuntimeError("private debug info")):
   r=self.client.post("/api/c4/release/evaluate",json={})
  self.assertEqual(r.status_code,500);self.assertNotIn("private debug info",r.text)
 def test_acceptance_uses_helper(self):
  with patch.object(self.route.service,"fixtures",return_value={"mode":"synthetic_acceptance"}) as helper:
   r=self.client.post("/api/c4/acceptance/verify",json={})
  self.assertEqual(r.status_code,200);helper.assert_called_once()
