import sys
from pathlib import Path
import unittest
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'scripts'))
from pairing_formation.pairing_channels import angular_model
from pairing_formation.dark_pairs import PairDarkSectors
from pairing_formation.gaussian_bins import angular_invariant_basis
from pairing_formation.projected_gaussian import torch_module
from integrate_dark_gaussians import coordinates


class DarkCoordinateTests(unittest.TestCase):
    def test_operator_and_gpu_formula_agree_on_cpu(self):
        torch=torch_module();torch.set_num_threads(1)
        model,forms=angular_model(10);dark=PairDarkSectors(model,forms)
        training={f'dark_m{m}':D for m,D in enumerate(dark.projectors)}
        rng=np.random.default_rng(947)
        atoms=rng.normal(size=(5,model.dim))+1j*rng.normal(size=(5,model.dim))
        actual=coordinates(torch.as_tensor(atoms),model,training,torch).numpy()
        for a,value in zip(atoms,actual):
            normalized=a/np.linalg.norm(a)
            expected=[float(abs(normalized)**2@model.m/model.r),
                      dark.evaluate(normalized)['multiple_transfer_either_dark']]
            np.testing.assert_allclose(value,expected,atol=2e-14)

    def test_labels_invariant_under_used_symmetries(self):
        torch=torch_module();torch.set_num_threads(1)
        model,forms=angular_model(10);dark=PairDarkSectors(model,forms)
        training={f'dark_m{m}':D for m,D in enumerate(dark.projectors)}
        _,permutations=angular_invariant_basis(model)
        rng=np.random.default_rng(948)
        atoms=rng.normal(size=(3,model.dim))+1j*rng.normal(size=(3,model.dim))
        original=coordinates(torch.as_tensor(atoms),model,training,torch).numpy()
        for p in permutations:
            changed=coordinates(torch.as_tensor(atoms[:,p]),model,training,torch).numpy()
            np.testing.assert_allclose(changed,original,atol=3e-14)


if __name__=='__main__':unittest.main()
