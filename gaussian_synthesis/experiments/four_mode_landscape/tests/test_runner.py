"""Runner boundaries: retained results, controlled stops, and solver isolation."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap  # noqa: F401
import numpy as np

import common
from gaussian_synthesis.solver import ResourceBudget
from reference import cat_state, cat_witness, quench_state, quench_reference


class RunnerTests(unittest.TestCase):
    def test_existing_output_is_rejected_without_changing_contents(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "preserved"
            output.mkdir()
            sentinel = output / "previous_result.txt"
            sentinel.write_text("Do not replace previous results.\n", encoding="utf-8")
            before = sentinel.read_bytes()
            with self.assertRaises(FileExistsError):
                common.run_experiment(common.ExperimentConfig("static"), output)
            self.assertEqual(sentinel.read_bytes(), before)
            self.assertEqual(list(output.iterdir()), [sentinel])

    def test_expired_budget_saves_failure_report_without_observations_or_confidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "stopped"
            config = common.ExperimentConfig("static")
            expired = ResourceBudget(config.max_seconds, config.max_memory_mib,
                                     started=time.monotonic() - config.max_seconds - 1)
            # Do not let an experiment replace the surrounding test runner's
            # process-wide watchdog; the actual cooperative budget expires.
            with patch.object(common, "ResourceBudget", return_value=expired), \
                    patch.object(common.signal, "signal"), \
                    patch.object(common.signal, "setitimer"), \
                    redirect_stdout(io.StringIO()):
                state = common.run_experiment(config, output)
            self.assertEqual(state["status"], "resource_limit")
            self.assertFalse(state["acceptance_passed"])
            self.assertEqual(state["completed_stages"], [])
            self.assertEqual(state["anchors"], [])
            self.assertFalse((output / "observations.npz").exists())
            for name in ("config.json", "summary.json", "report.md"):
                self.assertTrue((output / name).is_file(), name)
            saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, state)
            self.assertFalse(any("confidence" in key and key != "confidence_assumptions"
                                 for key in saved))
            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("未通过", report)
            self.assertIn("resource_limit", report)
            metadata = json.loads((output / "config.json").read_text(encoding="utf-8"))
            groups = metadata["rng_streams"]
            self.assertEqual(set(groups), {"observations", "reference", "control", "anchors"})
            keys = [tuple(stream["spawn_key"]) for streams in groups.values() for stream in streams]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertTrue(all(len(streams) == 4 for streams in groups.values()))
            self.assertAlmostEqual(metadata["reconstruction_delta_per_anchor"], .05 / 9)

    def test_anchor_calls_use_one_budget_and_preallocated_failure_probability(self):
        budget = ResourceBudget(180, 512)
        captured = []
        saves = []
        with tempfile.TemporaryDirectory() as temporary:
            for kind, count in (("static", 4), ("quench", 5)):
                with self.subTest(kind=kind):
                    config = common.ExperimentConfig(kind)
                    parameters = (np.linspace(0, .5, 51) if kind == "static"
                                  else np.linspace(0, 2*np.pi, 81))
                    if kind == "static":
                        targets = np.asarray([cat_state(r) for r in parameters])
                        references = np.asarray([cat_witness(r) for r in parameters])
                        rs = parameters
                    else:
                        targets = np.asarray([quench_state(phi) for phi in parameters])
                        mapped = [quench_reference(phi) for phi in parameters]
                        references = np.asarray([value.witness for value in mapped])
                        rs = np.asarray([value.r for value in mapped])
                    vectors = {"parameters": parameters, "psi": targets,
                               "u_reference": references, "r": rs}
                    # Select the independently derived reference corresponding
                    # to the supplied target only inside the fake result. The
                    # production call must not pass it into synthesize.
                    def fake_synthesize(target, *args, **kwargs):
                        captured.append((target.copy(), args, kwargs))
                        distances = np.linalg.norm(targets - target, axis=1)
                        witness = references[int(np.argmin(distances))]
                        stage = SimpleNamespace(solution=SimpleNamespace(u=witness),
                                                validation=SimpleNamespace(upper_bound=.03, samples=123))
                        saver = Mock(side_effect=lambda path: saves.append(Path(path)))
                        return SimpleNamespace(status="certified", stages=[stage], save=saver)
                    output = Path(temporary) / kind
                    output.mkdir()
                    state = {"anchors": []}
                    with patch.object(common, "synthesize", side_effect=fake_synthesize), \
                            redirect_stdout(io.StringIO()):
                        rows = common.run_anchors(config, vectors,
                                                  np.random.SeedSequence(771).spawn(count),
                                                  output, budget, state)
                    self.assertEqual(len(rows), count)
                    self.assertTrue(all(row["passed"] for row in rows))
                    self.assertEqual(state["anchors"], rows)
                    self.assertTrue((output / "anchors.csv").is_file())
            self.assertEqual(len(captured), 9)
            self.assertEqual(len(saves), 9)
            for _, args, kwargs in captured:
                self.assertEqual(args, ())
                self.assertEqual(set(kwargs), {"config", "budget"})
                self.assertIs(kwargs["budget"], budget)
                settings = kwargs["config"]
                self.assertEqual(settings.orders, (2,))
                self.assertAlmostEqual(settings.total_failure_probability, .05 / 9)
                self.assertEqual(settings.max_validation_samples, 12000000)


if __name__ == "__main__":
    unittest.main()
