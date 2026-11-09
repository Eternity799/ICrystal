"""Release-integrity and interface tests."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import release_runtime as runtime


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_local_manifest_integrity(self):
        manifest=runtime.verify_release()
        self.assertGreater(len(manifest['files']),20)
        self.assertTrue(all(not Path(row['path']).is_absolute() for row in manifest['files']))

    def test_modified_release_file_is_rejected(self):
        previous=runtime.ROOT
        try:
            with tempfile.TemporaryDirectory() as temp:
                runtime.ROOT=Path(temp)
                p=Path(temp)/'core.py'; p.write_text('x = 1\n',encoding='utf-8')
                runtime.write_json(Path(temp)/'release_manifest.json',dict(files=[dict(path='core.py',sha256=runtime.digest(p))]))
                runtime.verify_release()
                p.write_text('x = 2\n',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'modified'): runtime.verify_release()
        finally:
            runtime.ROOT=previous

    def test_frozen_baselines_share_protocol_and_saved_quality_definition(self):
        for target in ('main','target700'):
            cfg,grid,control,curve,domain,meta=runtime.load_inputs(target)
            with np.load(ROOT/'assets'/f'b3_{target}.npz',allow_pickle=False) as z:
                q=runtime.quality(z['counts_per_kg_water'],z['quality'][:,0],z['quality'][0,0],grid)
                np.testing.assert_allclose(q,z['quality'],rtol=1e-12,atol=1e-9)
                measured=runtime.score(z['quality'],curve,control,domain)['total']
            self.assertAlmostEqual(measured,meta['summary']['objective'],places=7)
            self.assertEqual(control['weights']['yield_terminal'],0.01)
            self.assertEqual(control['constraints']['supersaturation_floor'],0.99)

    def test_pretrained_model_shape_inventory_and_source_identity(self):
        predictor=runtime.load_predictor('cpu')
        with np.load(ROOT/'assets/example_history.npz',allow_pickle=False) as z:
            history={key:z[key].copy() for key in ('n','c','t')}
            total=float(z['initial_total']); future=z['future_t'].copy()
        output=predictor.rollout(history,future,total)
        self.assertEqual(output['counts'].shape,(4,200))
        self.assertTrue(output['valid_mask'].all())
        pm=predictor.model.particle_mass.detach().numpy()
        np.testing.assert_allclose(output['concentration']+output['counts']@pm,total,rtol=1e-13,atol=1e-13)
        self.assertTrue((output['counts']>=0).all())

    def test_external_streams_are_shared_bounded_prefix_stable(self):
        settings=dict(temperature_limit_c=1.,process_rho=.9,measurement_temperature_std_c=.1,
            measurement_concentration_relative_std=.005,minimum_kinetic_factor=.55,maximum_kinetic_factor=1.45)
        a=runtime.streams('combined',20261020,settings,180)
        b=runtime.streams('combined',20261020,settings,12)
        for full,short in zip(a,b): np.testing.assert_array_equal(full[:len(short)],short)
        self.assertEqual(a[1][0],0.)
        self.assertLessEqual(np.max(np.abs(a[1])),1.)
        self.assertTrue(((a[2]>=.55)&(a[2]<=1.45)).all())
        nominal=runtime.streams('nominal',20261020,settings,12)
        self.assertTrue((nominal[0]==0).all() and (nominal[1]==0).all() and (nominal[2]==1).all())

    def test_schema_status_compatibility_and_unknown_status_rejection(self):
        from surrogate.core.data import read_manifest
        from surrogate.core.mpc.data import _validate_manifest
        legacy='\u5b8c\u6574CSD\u6570\u636e\u751f\u6210\u5b8c\u6210_\u672a\u8bad\u7ec3'
        with tempfile.TemporaryDirectory() as temp:
            for status in ('dataset_complete_not_trained',legacy):
                payload=dict(schema_version=1,status=status,plan=[],completed=[])
                runtime.write_json(Path(temp)/'manifest.json',payload)
                self.assertEqual(read_manifest(temp)['status'],status)
                _validate_manifest(payload)
            runtime.write_json(Path(temp)/'manifest.json',dict(schema_version=1,status='partial',plan=[],completed=[]))
            with self.assertRaises(ValueError): read_manifest(temp)

    def test_invalid_prediction_quality_stays_unavailable(self):
        _,grid,*_=runtime.load_inputs()
        q=runtime.quality(np.full((1,200),np.nan),np.array([.1]),.15,grid)
        self.assertTrue(np.isnan(q).all())

    def test_new_training_directory_can_be_selected_without_bundled_export_metadata(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            metadata=dict(status='control_oriented_training_complete_pending_independent_prediction_and_control',
                workflow='mpc_oriented_training_v2',fingerprint=dict(source='fixture'))
            runtime.write_json(Path(temp)/'run.json',metadata)
            marker=SimpleNamespace(saved=dict(fingerprint=metadata['fingerprint']))
            with patch('surrogate.core.mpc.inference.MPCPredictor.from_run',return_value=marker) as loader:
                self.assertIs(runtime.load_predictor('cpu',model_run=Path(temp)),marker)
                loader.assert_called_once_with(Path(temp).resolve(),'best','cpu')

    def test_running_new_model_is_rejected_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime.write_json(Path(temp)/'run.json',dict(status='control_oriented_training_running_unvalidated'))
            with patch('surrogate.core.mpc.inference.MPCPredictor.from_run') as loader:
                with self.assertRaisesRegex(ValueError,'complete|running|incomplete'):
                    runtime.load_predictor('cpu',model_run=Path(temp))
                loader.assert_not_called()

    def test_json_output_accepts_portable_path_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'metadata.json'
            runtime.write_json(path,dict(input=Path('data')/'labels',arguments=[Path('model')]))
            result=json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(result['input'],str(Path('data')/'labels'))

    def test_new_offline_export_roundtrips_and_rejects_changed_candidate_files(self):
        _,_,control,curve,_,meta=runtime.load_inputs('main')
        with np.load(ROOT/'assets/b3_main.npz',allow_pickle=False) as z:
            arrays={key:z[key].copy() for key in z.files}
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            runtime.write_baseline(folder,arrays,control,meta['summary'],'search_completed')
            result=runtime.load_inputs('main',b3_run=folder)
            np.testing.assert_array_equal(result[3],curve)
            self.assertEqual(result[5]['summary']['objective'],meta['summary']['objective'])
            with (folder/'frozen_schedule.json').open('a',encoding='utf-8') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError,'baseline.*modified'):
                runtime.load_inputs('main',b3_run=folder)


if __name__=='__main__':
    unittest.main()
