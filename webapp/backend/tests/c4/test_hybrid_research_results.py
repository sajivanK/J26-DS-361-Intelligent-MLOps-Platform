"""Saved-summary integration checks. All test research numbers are synthetic."""
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

BACKEND = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('hybrid_reader_under_test',BACKEND/'app/components/c4_hybrid_research.py')
reader = importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)


def digest(data): return hashlib.sha256(data).hexdigest()


def document(directory,stem,payload):
    directory.mkdir(parents=True,exist_ok=True)
    data = (json.dumps(payload,sort_keys=True,allow_nan=False)+'\n').encode()
    (directory/(stem+'.json')).write_bytes(data)
    (directory/(stem+'.sha256')).write_text(digest(data)+'\n')
    return digest(data)


def table(rows):
    stream = io.StringIO(newline=''); writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader(); writer.writerows(rows); return stream.getvalue().encode()


class HybridResearchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        if os.name == 'nt':
            self.root = Path('\\\\?\\'+str(self.root))
            self.addCleanup(lambda:shutil.rmtree(self.root) if self.root.exists() else None)
        self.bundle = self.root/'c4_artifacts/forecasting_experiments/adult'/reader.VERSION
        self.output = self.bundle/'outer-future-evaluator-v1/completion-v1'
        protocol_hash = document(self.bundle,'validation_protocol',dict(primary_method=reader.PRIMARY))
        pin = patch.object(reader,'EXPECTED_PROTOCOL_SHA256',protocol_hash); pin.start(); self.addCleanup(pin.stop)
        models = {m:dict(frozen='fixture') for m in reader.METHODS}
        n2_hash = document(self.bundle/'n2-calibration-v1/completion-v1','calibration',dict(model_bindings=models))
        selected = dict(model_bindings=models,N2_calibration_sha256=n2_hash,production_release_authorized=False,
            context_support=dict(unsupported_contexts=['LOW','HIGH']),methods={m:{f:dict(status='UNSUPPORTED_SELECTION_REVIEW_ONLY',selected_policy=dict(kind='review_only',policy_id='unsupported-review-only')) for f in ('global','adaptive')} for m in models})
        policy_hash = document(self.bundle/'policy-selection-v1/completion-v1','selection',selected)
        index = [dict(scenario_id=f's{i}',group_id=f'g{i%11}',split_role='outer_test') for i in range(56)]
        self.claim = dict(purpose='fresh_outer_external_future_evaluation_v1',scenario_ids=[r['scenario_id'] for r in index],
            production_release_authorized=False,selected_policy_decisions_frozen_before_future=True,future_evaluation_authorized=True,
            expected_cohort_rows=12899,model_bindings=models,N2_calibration_sha256=n2_hash,policy_selection_sha256=policy_hash,
            **dict.fromkeys(('candidate_training_authorized','generator_refitting_authorized','N1_refitting_authorized','N2_fitting_authorized','policy_fitting_authorized'),False))
        claim_hash = document(self.output.parent,'claim',self.claim)
        coverage = dict(N2_calibration_sha256=n2_hash,N2_refitted=False,methods={m:dict(scenario_joint_coverage=1.,complete_group_joint_coverage=1.,targets={t:dict(equal_group_weighted_mean_width_fraction=.02,equal_group_weighted_interval_score_alpha_0_05=.02) for t in reader.TARGETS}) for m in models})
        self.governance = dict(policy_selection_sha256=policy_hash,policy_refitted=False,formal_conditional_risk_guarantee_established=False,
            production_release_authorized=False,admitted_cases=56,blocked_cases_retained=536,assigned_groups=16,nonempty_groups=11,all_blocked_groups=5,methods={})
        signals=[]
        for m in models:
            self.governance['methods'][m]={}
            for family in ('global','adaptive'):
                counts=dict(AUTO_DEPLOY=0,HUMAN_REVIEW=47 if 'linear' in m else 56,ESCALATE=9 if 'linear' in m else 0)
                self.governance['methods'][m][family]=dict(selected_policy=selected['methods'][m][family]['selected_policy'],statistics=dict(decision_counts=counts,equal_group_automation=0.,equal_group_conditional_unsafe_rate=None))
                signals.extend(dict(**r,method=m,family=family,decision='ESCALATE' if 'linear' in m and i<9 else 'HUMAN_REVIEW') for i,r in enumerate(index))
        self.measured = [dict(method=m,target=t,mae=.001,rmse=.002,mean_pinball=.0003,MAE_pp=.1,RMSE_pp=.2,Pinball_pp=.03,r2=.25) for m in models for t in reader.TARGETS]
        files={'case_index.csv':table(index),'selected_policy_decisions.csv':table(signals),'forecast_metrics.csv':table(self.measured),
            'N2_outer_diagnostics.json':json.dumps(coverage).encode(),'policy_diagnostics.json':json.dumps(self.governance).encode()}
        self.report=dict(purpose='fresh_outer_future_evaluation_complete',evaluator_claim_sha256=claim_hash,cases=56,assigned_groups=16,nonempty_groups=11,blocked_cases_retained=536,
            production_release_authorized=False,training_executed=False,policy_refitted=False,N2_calibrated=True,policy_selection_frozen=True,
            independent_dataset_validation=False,held_out_scenario_group_evaluation=True,selected_policy_decisions_frozen_before_future=True,
            forecasts_frozen_before_future=True,N2_calibration_sha256=n2_hash,policy_selection_sha256=policy_hash,records=index,files={})
        self.output.mkdir(parents=True,exist_ok=True)
        for name,data in files.items():
            (self.output/name).write_bytes(data);self.report['files'][name]=digest(data)
        self.save_report()

    def save_report(self): document(self.output,'future_batch',self.report)

    def replace_output(self,name,data):
        (self.output/name).write_bytes(data);self.report['files'][name]=digest(data);self.save_report()

    def test_verified_summary_returns_all_methods_and_pp_units(self):
        result=reader.read_results(self.root)
        self.assertEqual(len(result['methods']),4);self.assertEqual(result['counts']['nonempty_groups'],11)
        self.assertTrue(result['evidence_verified']);self.assertFalse(result['inference_executed'])
        self.assertEqual(result['methods'][0]['metrics'][0]['mae_pp'],.1)
        self.assertEqual(result['methods'][0]['metrics'][0]['width_pp'],2.)
        self.assertEqual(result['methods'][2]['policies']['adaptive']['decision_counts']['ESCALATE'],9)
        self.assertIsNone(result['methods'][0]['policies']['global']['conditional_unsafe_rate'])
        self.assertNotIn(str(self.root),json.dumps(result))

    def test_read_only_no_model_loading_or_writes(self):
        before={str(p):digest(p.read_bytes()) for p in self.root.rglob('*') if p.is_file()}
        with patch.dict(sys.modules,{'lightgbm':None,'xgboost':None,'torch':None}):reader.read_results(self.root)
        after={str(p):digest(p.read_bytes()) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_missing_environment(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(reader.ResearchUnavailable):reader.read_results()

    def test_missing_artifact(self):
        (self.output/'forecast_metrics.csv').unlink()
        with self.assertRaises(reader.ResearchUnavailable):reader.read_results(self.root)

    def test_tampered_output_hash(self):
        (self.output/'forecast_metrics.csv').write_bytes(b'changed')
        with self.assertRaises(reader.ResearchUnavailable):reader.read_results(self.root)

    def test_wrong_protocol(self):
        with patch.object(reader,'EXPECTED_PROTOCOL_SHA256','0'*64):
            with self.assertRaises(reader.ResearchUnavailable):reader.read_results(self.root)

    def test_wrong_report_role_counts_and_release_scope(self):
        for key,value in [('cases',55),('nonempty_groups',16),('production_release_authorized',True),('policy_refitted',True),('independent_dataset_validation',True)]:
            with self.subTest(key=key):
                old=self.report[key];self.report[key]=value;self.save_report()
                with self.assertRaises(reader.ResearchUnavailable):reader.read_results(self.root)
                self.report[key]=old;self.save_report()

    def test_path_traversal_rejected(self):
        self.report['files']['../secret']='0'*64;self.save_report()
        with self.assertRaises(reader.ResearchUnavailable):reader.read_results(self.root)

    def test_duplicate_or_missing_metrics_rejected(self):
        for rows in (self.measured[:-1],self.measured[:-1]+[self.measured[0]]):
            self.replace_output('forecast_metrics.csv',table(rows))
            with self.assertRaises(reader.ResearchUnavailable):reader.read_results(self.root)

    def test_negative_r2_is_valid(self):
        self.measured[0]['r2']=-.4;self.replace_output('forecast_metrics.csv',table(self.measured))
        self.assertEqual(reader.read_results(self.root)['methods'][0]['metrics'][0]['r2'],-.4)

    def test_nonfinite_or_wrong_units_rejected(self):
        for value in ('nan',100):
            self.measured[0]['MAE_pp']=value;self.replace_output('forecast_metrics.csv',table(self.measured))
            with self.assertRaises(reader.ResearchUnavailable):reader.read_results(self.root)

    def test_wrong_n2_link_rejected(self):
        self.report['N2_calibration_sha256']='0'*64;self.save_report()
        with self.assertRaises(reader.ResearchUnavailable):reader.read_results(self.root)

    def test_wrong_policy_link_rejected(self):
        self.report['policy_selection_sha256']='0'*64;self.save_report()
        with self.assertRaises(reader.ResearchUnavailable):reader.read_results(self.root)

    def test_zero_deployment_risk_cannot_be_reported_as_zero(self):
        self.governance['methods'][reader.PRIMARY]['global']['statistics']['equal_group_conditional_unsafe_rate']=0.
        self.replace_output('policy_diagnostics.json',json.dumps(self.governance).encode())
        with self.assertRaises(reader.ResearchUnavailable):reader.read_results(self.root)

    def test_decision_counts_must_match_saved_signals(self):
        self.governance['methods'][reader.PRIMARY]['global']['statistics']['decision_counts']['HUMAN_REVIEW']=55
        self.replace_output('policy_diagnostics.json',json.dumps(self.governance).encode())
        with self.assertRaises(reader.ResearchUnavailable):reader.read_results(self.root)

    def test_error_does_not_expose_local_paths(self):
        (self.output/'forecast_metrics.csv').unlink()
        with self.assertRaises(reader.ResearchUnavailable) as caught:reader.read_results(self.root)
        self.assertNotIn(str(self.root),str(caught.exception))


try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
except ImportError:
    FastAPI=None


@unittest.skipUnless(FastAPI,'FastAPI required for routing checks')
class HybridRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib
        for name,path in [('c4_hybrid_app',BACKEND/'app'),('c4_hybrid_app.components',BACKEND/'app/components')]:
            mod=types.ModuleType(name);mod.__path__=[str(path)];sys.modules[name]=mod
        cls.route=importlib.import_module('c4_hybrid_app.routers.c4')
        app=FastAPI();app.include_router(cls.route.router);cls.client=TestClient(app)

    def test_new_route_forwards_separately(self):
        with patch.object(self.route.service,'hybrid_results',return_value={'version':reader.VERSION}) as call:
            response=self.client.get('/api/c4/research-results/hybrid')
        self.assertEqual(response.status_code,200);self.assertEqual(response.json()['version'],reader.VERSION);call.assert_called_once()

    def test_unavailable_revised_artifacts_do_not_return_metrics(self):
        with patch.object(self.route.service,'hybrid_results',side_effect=self.route.service.C4Unavailable('Unavailable')):
            response=self.client.get('/api/c4/research-results/hybrid')
        self.assertEqual(response.status_code,503);self.assertNotIn('methods',response.json())

    def test_worker_forwarding_uses_get(self):
        opener=MagicMock();opener.open.return_value.__enter__.return_value.read.return_value=b'{"version":"fixture"}'
        with patch.object(self.route.service,'build_opener',return_value=opener):self.route.service.hybrid_results()
        request=opener.open.call_args.args[0]
        self.assertEqual(request.get_method(),'GET');self.assertTrue(request.full_url.endswith('/api/c4/research-results/hybrid'))

    def test_original_route_is_not_redirected(self):
        with patch.object(self.route.service,'recorded_results',return_value={'original':True}),patch.object(self.route.service,'hybrid_results',side_effect=AssertionError('must remain separate')):
            self.assertEqual(self.client.get('/api/c4/research-results').json(),{'original':True})


@unittest.skipUnless(FastAPI,'FastAPI required for worker checks')
class HybridWorkerTests(unittest.TestCase):
    def setUp(self):
        class Unavailable(RuntimeError): pass
        self.service=types.SimpleNamespace(C4Unavailable=Unavailable,hybrid_results=MagicMock(return_value={'version':reader.VERSION}))
        parent=types.ModuleType('app');parent.__path__=[]
        components=types.ModuleType('app.components');components.c4_service=self.service
        spec=importlib.util.spec_from_file_location('isolated_hybrid_worker',BACKEND/'c4_worker.py')
        worker=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'app':parent,'app.components':components}):spec.loader.exec_module(worker)
        self.client=TestClient(worker.app)

    def test_worker_serves_new_summary_without_controller(self):
        response=self.client.get('/api/c4/research-results/hybrid')
        self.assertEqual(response.status_code,200);self.assertEqual(response.json()['version'],reader.VERSION)
        self.service.hybrid_results.assert_called_once()

    def test_worker_returns_503_for_missing_new_evidence(self):
        self.service.hybrid_results.side_effect=self.service.C4Unavailable('missing')
        self.assertEqual(self.client.get('/api/c4/research-results/hybrid').status_code,503)


@unittest.skipUnless(FastAPI,'Package namespace used by service integration checks')
class HybridServiceBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib
        for name,path in [('c4_bridge_app',BACKEND/'app'),('c4_bridge_app.components',BACKEND/'app/components')]:
            mod=types.ModuleType(name);mod.__path__=[str(path)];sys.modules[name]=mod
        cls.service=importlib.import_module('c4_bridge_app.components.c4_service')
        cls.reader=importlib.import_module('c4_bridge_app.components.c4_hybrid_research')

    def test_bridge_does_not_load_models(self):
        with patch.object(self.reader,'read_results',return_value={'saved':True}) as call,patch.object(self.service,'_runtime',side_effect=AssertionError('No model loading')):
            self.assertEqual(self.service.hybrid_results(),{'saved':True});call.assert_called_once()

    def test_bridge_maps_unavailable_into_existing_service_error(self):
        with patch.object(self.reader,'read_results',side_effect=self.reader.ResearchUnavailable('missing')):
            with self.assertRaises(self.service.C4Unavailable):self.service.hybrid_results()
