import unittest
import numpy as np
from fk.dictionaries import Dictionary
from fk.validation import gaussian_probe_reference,formal_budget,ReconstructionAccumulator
from fk.runner import validate_config,run,ROOT
from fk.distributions import CoherentBins
import json
import tempfile
import signal
import time
from fk.runtime import Budget,ResourceLimit
from fk.pilot import calibrate
from fk.model import build_model


class ValidationTests(unittest.TestCase):
    def test_gaussian_endpoint_and_budget(self):
        d = Dictionary('full')
        self.assertAlmostEqual(gaussian_probe_reference(d,8)['mass'],2.6989986640710395,places=12)
        b = formal_budget(d,2,1.,.05/720,.005,65536)
        self.assertEqual(b['status'],'validation_budget_exceeded')
        self.assertGreater(b['required_samples'],65536)

    def test_residual_decomposition_and_phase_invariance(self):
        rng = np.random.default_rng(19)
        g = Dictionary().sample(32,rng).kets
        u = g[0]
        S = np.eye(128,3,dtype=complex)
        a,b = [ReconstructionAccumulator(Dictionary(),[u],[2]) for _ in range(2)]
        a.add(g)
        b.add(g*np.exp(1j*rng.uniform(-np.pi,np.pi,size=32))[:,None])
        ra,rb = a.result(0,u,S),b.result(0,u,S)
        np.testing.assert_allclose(ra['reconstruction'],rb['reconstruction'],atol=1e-13)
        self.assertAlmostEqual(ra['sampled_full_residual']**2,
                               ra['validation_singlet_residual']**2+ra['validation_outside_singlet_norm']**2,places=12)

    def test_no_overwrite_and_budget_config(self):
        config=json.loads((ROOT/'configs/pilot.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError): run(directory,config)
        config['sample_sizes']=[131072]
        with self.assertRaises(ValueError): validate_config(config)

    def test_expired_budget_does_not_arm_alarm(self):
        before = signal.getsignal(signal.SIGALRM)
        with self.assertRaises(ResourceLimit):
            with Budget(total_deadline=time.monotonic()-1):
                self.fail('Expired budget admitted work')
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL),(0.,0.))
        self.assertEqual(signal.getsignal(signal.SIGALRM),before)

    def test_calibration_partial_saved_without_certificate(self):
        config=json.loads((ROOT/'configs/pilot.json').read_text())
        config.update(sample_sizes=[256,512],orders=[1])
        snapshots=[]
        class InterruptedBudget:
            calls=0
            def check(self):
                self.calls+=1
                if self.calls>1: raise ResourceLimit('test interruption')
        with self.assertRaises(ResourceLimit):
            calibrate(build_model(),'full',19,config,InterruptedBudget(),
                      lambda report,arrays:snapshots.append((dict(report),dict(arrays))))
        self.assertEqual(snapshots[0][0]['completed_samples'],256)
        self.assertFalse(snapshots[0][0]['formally_certified'])
        self.assertEqual(int(snapshots[0][1]['count']),256)
