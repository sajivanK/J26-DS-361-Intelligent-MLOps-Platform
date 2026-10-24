"""Revised replay integrity, pre-future scope and HTTP isolation tests."""
import copy
import hashlib
import importlib
import importlib.util
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
spec = importlib.util.spec_from_file_location('revised_reader_test', BACKEND/'app/components/c4_revised_case.py')
reader = importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)


class RevisedReplayTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(('\\\\?\\'+str(self.root.resolve())) if os.name == 'nt' else self.root))
        self.seed = json.loads((Path(__file__).parent/'fixtures/revised_case_seed.json').read_bytes())
        self.bundle = self.root/'c4_artifacts/forecasting_experiments/adult'/reader.VERSION
        self.run = self.root/'c4_artifacts/benchmark_runs/adult'/reader.VERSION/reader.SCENARIO_ID
        self.build()

    def doc(self, directory, stem, value):
        directory.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
        digest = hashlib.sha256(data).hexdigest()
        (directory/(stem+'.json')).write_bytes(data)
        (directory/(stem+'.sha256')).write_text(digest+'\n', encoding='utf-8')
        return digest

    def build(self):
        s = self.seed
        ph = self.doc(self.bundle,'validation_protocol',{'primary_method':reader.METHODS['hybrid']})
        bindings = {m:{'fixture':'fixed'} for m in reader.METHODS.values()}
        nh = self.doc(self.bundle/'n2-calibration-v1/completion-v1','calibration',{'methods':s['calibrations'],'model_bindings':bindings})
        policies = {m:{f:{'selected_policy':s['signals']['methods'][m]['policies'][f]['selected_policy'],'status':s['signals']['methods'][m]['policies'][f]['selection_status']} for f in ('global','adaptive')} for m in reader.METHODS.values()}
        sh = self.doc(self.bundle/'policy-selection-v1/completion-v1','selection',{'methods':policies,'model_bindings':bindings,'N2_calibration_sha256':nh})
        ev = copy.deepcopy(s['evidence'])
        eh = self.doc(self.root/'c4_artifacts/pre_release/adult'/reader.SCENARIO_ID,'evidence',ev)
        md = dict(scenario_id=reader.SCENARIO_ID,model_sha256=ev['candidate_model_sha256'],champion_model_sha256=ev['champion_model_sha256'],artifact_version=ev['candidate_version'],champion_version=ev['champion_version'],repair_csv_sha256=ev['repair_csv_sha256'])
        mh = self.doc(self.run/'scoped_candidate_validation-v1','metadata',md)
        vh = self.doc(self.run/'scoped_candidate_validation-v1','validation',dict(scenario=s['scenario'],pre_release_evidence_sha256=eh,candidate_metadata_sha256=mh,candidate_reload_verified=True,future_evaluation_executed=False))
        uh = self.doc(self.run/'pre_release_freeze-v1','upstream_simulation',dict(upstream=s['upstream'],scope='controlled_offline_simulation',production_package_approved=False,source_declarations={'c1':'mock_oracle','c2':'simulated','c3':'offline_simulated'},privacy_interpretation='Exploratory simulation only.'))
        snap = dict(scenario=s['scenario'],complete_decision_evidence_frozen=True,future_evaluation_executed=False,release_authorized=False,production_package_approved=False,upstream_simulation_sha256=uh,pre_release_evidence_sha256=eh,created_at_utc='fixture-time',runtime_checks={'schema_compatible':True,'rollback_ready':True,'rollback_scope':'offline_artifact_restore'},dependencies={'/external/scoped_candidate_validation-v1/metadata.json':mh,'/external/scoped_candidate_validation-v1/validation.json':vh})
        dh = self.doc(self.run/'pre_release_freeze-v1','freeze',snap)
        values = reader.feature_row(s['upstream'],ev)
        features = dict(scenario_id=reader.SCENARIO_ID,columns=list(values),values=list(values.values()),basis='pre_release_diagnosed_populations_only_no_future_targets')
        state = dict(scenario=s['scenario'],decision_freeze_sha256=dh,N2_calibration_sha256=nh,policy_selection_sha256=sh,model_bindings=bindings,forecasts_frozen_before_future=True,selected_policy_decisions_frozen_before_future=True,future_evaluation_executed=False,production_release_authorized=False)
        for stem,key in [('features','features_sha256'),('forecasts','forecasts_sha256'),('controller_decisions','controller_decisions_sha256')]:
            obj = features if stem=='features' else s['forecasts'] if stem=='forecasts' else s['signals']
            state[key]=self.doc(self.run/'forecast_freeze-v1',stem,obj)
        fh = self.doc(self.run/'forecast_freeze-v1','freeze',state)
        record = dict(scenario_id=reader.SCENARIO_ID,split_role='outer_test',forecast_freeze_sha256=fh,decision_freeze_sha256=dh,N2_calibration_sha256=nh,policy_selection_sha256=sh,feature_sha256=state['features_sha256'],forecasts_sha256=state['forecasts_sha256'],controller_decisions_sha256=state['controller_decisions_sha256'])
        records=[record]+[dict(scenario_id='zz-fixture-%02d'%i) for i in range(55)]
        bh = self.doc(self.bundle/'outer-forecast-freezer-v1/completion-v1','forecast_batch',dict(records=records,cases=56,forecasts_frozen_before_future=True,selected_policy_decisions_frozen_before_future=True,future_evaluation_executed=False,production_release_authorized=False,N2_calibration_sha256=nh,policy_selection_sha256=sh,model_bindings=bindings))
        self.hashes=(ph,bh)

    def replay(self, **kwargs):
        with patch.object(reader,'PROTOCOL_SHA256',self.hashes[0]),patch.object(reader,'BATCH_SHA256',self.hashes[1]):return reader.replay(data_root=self.root,**kwargs)

    def test_primary_replay(self):
        r=self.replay();self.assertEqual(r['result']['decision'],'HUMAN_REVIEW');self.assertEqual(r['result']['method'],'hybrid')
        self.assertEqual(sum(d['status']=='SAFE' for d in r['result']['n3']['dimensions'].values()),1)
        self.assertTrue(all(r['hard_gates'].values()));self.assertFalse(r['inference_executed']);self.assertTrue(r['result']['N1_replayed'])
        self.assertFalse(r['result']['N1_executed']);self.assertFalse(r['result']['production_release_authorized'])
        self.assertEqual(r['candidate_evaluation']['global']['candidate']['accuracy'],.8571428571428571)
        self.assertEqual(len(r['decision_time_features']),75)

    def test_all_methods_and_families_use_matched_forecasts(self):
        for method,model in reader.METHODS.items():
            for family in ('adaptive','global'):
                r=self.replay(method=method,family=family)
                self.assertEqual(r['result']['n1_forecasts'],self.seed['forecasts'][model]['processed'])
                self.assertEqual(r['n2_calibration'],self.seed['calibrations'][model]);self.assertEqual(r['result']['family'],family)

    def test_missing_environment(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(reader.RevisedCaseUnavailable):reader.replay()

    def test_invalid_requests_before_artifact_access(self):
        for kwargs,error in [({'case_id':'unknown'},KeyError),({'method':'unknown'},ValueError),({'family':'unknown'},ValueError)]:
            with patch.object(Path,'read_bytes',side_effect=AssertionError('No artifact reads')):
                with self.assertRaises(error):reader.replay(**kwargs)

    def test_tampered_forecast(self):
        (self.run/'forecast_freeze-v1/forecasts.json').write_text('{}')
        with self.assertRaises(reader.RevisedCaseUnavailable):self.replay()

    def test_missing_evidence_and_private_error(self):
        (self.run/'pre_release_freeze-v1/upstream_simulation.json').unlink()
        with self.assertRaises(reader.RevisedCaseUnavailable) as caught:self.replay()
        self.assertNotIn(str(self.root),str(caught.exception))

    def test_wrong_protocol_anchor(self):
        self.hashes=('0'*64,self.hashes[1])
        with self.assertRaises(reader.RevisedCaseUnavailable):self.replay()

    def test_wrong_batch_anchor(self):
        self.hashes=(self.hashes[0],'0'*64)
        with self.assertRaises(reader.RevisedCaseUnavailable):self.replay()

    def test_wrong_n3_status_even_with_updated_fixture_hashes(self):
        self.seed['signals']['methods'][reader.METHODS['hybrid']]['N3']['dimensions']['future_global_margin']['status']='SAFE';self.build()
        with self.assertRaises(reader.RevisedCaseUnavailable):self.replay()

    def test_wrong_n2_adjustment(self):
        self.seed['calibrations'][reader.METHODS['hybrid']]['adjustment']=0.;self.build()
        with self.assertRaises(reader.RevisedCaseUnavailable):self.replay()

    def test_wrong_policy_decision(self):
        self.seed['signals']['methods'][reader.METHODS['hybrid']]['policies']['adaptive']['result']['decision']='AUTO_DEPLOY';self.build()
        with self.assertRaises(reader.RevisedCaseUnavailable):self.replay()

    def test_wrong_gate(self):
        self.seed['signals']['hard_gates']['privacy_check']=False;self.build()
        with self.assertRaises(reader.RevisedCaseUnavailable):self.replay()

    def test_future_targets_rejected(self):
        self.seed['signals']['future_targets_used']=True;self.build()
        with self.assertRaises(reader.RevisedCaseUnavailable):self.replay()

    def test_read_only_no_models_or_future_reads(self):
        original=Path.read_bytes
        def guarded(path):
            self.assertIn(path.suffix,('.json','.sha256'));self.assertNotIn('future-evaluator',str(path));self.assertNotIn('external_future',str(path));return original(path)
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*') if p.is_file()}
        with patch.object(Path,'read_bytes',guarded),patch.dict(sys.modules,{'xgboost':None,'lightgbm':None,'torch':None}):self.replay()
        after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)


from fastapi import FastAPI
from fastapi.testclient import TestClient


class RevisedRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name,path in [('c4_revised_app',BACKEND/'app'),('c4_revised_app.components',BACKEND/'app/components')]:
            module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
        cls.route=importlib.import_module('c4_revised_app.routers.c4');app=FastAPI();app.include_router(cls.route.router);cls.client=TestClient(app)

    def test_defaults_forward_to_revised_service(self):
        with patch.object(self.route.service,'revised_evaluate',return_value={'mode':'revised'}) as call:r=self.client.post('/api/c4/release/revised/evaluate',json={})
        self.assertEqual(r.status_code,200);call.assert_called_once_with('revised-outer-case','hybrid','adaptive')

    def test_original_endpoint_preserved(self):
        with patch.object(self.route.service,'evaluate',return_value={'original':True}) as call,patch.object(self.route.service,'revised_evaluate',side_effect=AssertionError('Original must remain separate')):
            self.assertEqual(self.client.post('/api/c4/release/evaluate',json={}).status_code,200)
        call.assert_called_once_with('real-outer-case','lightgbm','adaptive')

    def test_bad_method_and_extra_future_target_rejected(self):
        for payload in ({'method':'unknown'},{'future_global_margin':.1}):self.assertEqual(self.client.post('/api/c4/release/revised/evaluate',json=payload).status_code,422)

    def test_unavailable_and_unknown_case(self):
        for error,code in [(self.route.service.C4Unavailable('Unavailable'),503),(KeyError('unknown'),404)]:
            with patch.object(self.route.service,'revised_evaluate',side_effect=error):self.assertEqual(self.client.post('/api/c4/release/revised/evaluate',json={}).status_code,code)

    def test_forwarding_post_payload(self):
        opener=MagicMock();opener.open.return_value.__enter__.return_value.read.return_value=b'{"evidence_verified":true}'
        with patch.object(self.route.service,'build_opener',return_value=opener):self.route.service.revised_evaluate()
        request=opener.open.call_args.args[0];self.assertEqual(request.get_method(),'POST');self.assertTrue(request.full_url.endswith('/release/revised/evaluate'))
        self.assertEqual(json.loads(request.data),{'case_id':'revised-outer-case','method':'hybrid','family':'adaptive'})


class RevisedWorkerTests(unittest.TestCase):
    def test_worker_endpoint_and_validation(self):
        class Unavailable(RuntimeError):pass
        service=types.SimpleNamespace(C4Unavailable=Unavailable,revised_evaluate=MagicMock(return_value={'evidence_verified':True}))
        parent=types.ModuleType('app');parent.__path__=[];components=types.ModuleType('app.components');components.c4_service=service
        spec=importlib.util.spec_from_file_location('isolated_revised_worker',BACKEND/'c4_worker.py');worker=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'app':parent,'app.components':components}):spec.loader.exec_module(worker)
        client=TestClient(worker.app);self.assertEqual(client.post('/api/c4/release/revised/evaluate',json={}).status_code,200)
        service.revised_evaluate.assert_called_once_with('revised-outer-case','hybrid','adaptive')
        self.assertEqual(client.post('/api/c4/release/revised/evaluate',json={'future_target':0}).status_code,422)

    def test_service_bridge(self):
        for name,path in [('c4_revised_bridge',BACKEND/'app'),('c4_revised_bridge.components',BACKEND/'app/components')]:
            module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
        service=importlib.import_module('c4_revised_bridge.components.c4_service');module=importlib.import_module('c4_revised_bridge.components.c4_revised_case')
        with patch.object(module,'replay',return_value={'replay':True}) as call,patch.object(service,'_runtime',side_effect=AssertionError('No original model load')):
            self.assertEqual(service.revised_evaluate(),{'replay':True});call.assert_called_once_with('revised-outer-case','hybrid','adaptive')
        with patch.object(module,'replay',side_effect=module.RevisedCaseUnavailable('missing')):
            with self.assertRaises(service.C4Unavailable):service.revised_evaluate()


if __name__=='__main__':unittest.main()
