import copy
import json
from pathlib import Path
import tempfile
import unittest
from fk3.runner import validate_config,aggregate_gates,run

ROOT=Path(__file__).resolve().parents[1]


class RunnerGates(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'configs/canonical.json').read_text())

    def test_scope_and_joint_accounting(self):
        validate_config(self.config)
        for key,value in [('planned_statistical_events',119),('final_samples',131072),('stream_namespace',0)]:
            cfg=dict(self.config,**{key:value})
            with self.assertRaises(ValueError):
                validate_config(cfg)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(FileExistsError):
                run(name,self.config)

    def test_missing_seed_and_inaccurate_group_cannot_pass(self):
        cfg=self.config
        row={'id':'t0_k2','t':0,'k':2,'n':65536,'is_probe':False,
             'mean_eta':.5,'ess':2000,'local_reconstruction_passed':True,
             'histogram_tv_previous':.01,'coherent_closure_residual':.001,
             'CQ_error':.001,'W_error':.001,'group_rmse':[.001]*3,
             'group_half_disagreement':[.001]*3}
        report={'rows':[row]+[dict(row,id=f'probe_k{k}',is_probe=True,probe_passed=True) for k in (2,4)]}
        allseeds={s:copy.deepcopy(report) for s in cfg['seeds']}
        good=aggregate_gates(allseeds,allseeds,cfg)['rows'][0]
        self.assertTrue(good['coherent_interpretation_allowed'])
        final=copy.deepcopy(allseeds)
        del final[cfg['seeds'][0]]
        self.assertFalse(aggregate_gates(allseeds,final,cfg)['rows'][0]['distribution_interpretation_allowed'])
        final=copy.deepcopy(allseeds)
        final[cfg['seeds'][0]]['rows'][0]['group_rmse'][0]=.03
        gate=aggregate_gates(allseeds,final,cfg)['rows'][0]
        self.assertTrue(gate['distribution_interpretation_allowed'])
        self.assertFalse(gate['coherent_interpretation_allowed'])
        final=copy.deepcopy(allseeds)
        final[cfg['seeds'][0]]['rows'][2]['id']='probe_k2'
        self.assertFalse(aggregate_gates(allseeds,final,cfg)['probe_calibration_passed'])
        development=copy.deepcopy(allseeds)
        development[0]=development.pop(cfg['seeds'][0])
        self.assertFalse(aggregate_gates(development,allseeds,cfg)['rows'][0]['reconstruction_passed'])


if __name__=='__main__':
    unittest.main()
