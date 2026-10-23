import copy
import json
from pathlib import Path
import tempfile
import unittest
from fk2.runner import validate_config, file_hashes, run

ROOT=Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'configs/followup.json').read_text())

    def test_protocol_caps_and_new_streams(self):
        validate_config(self.config)
        for key,bad in [('orbit_sizes',[129]),('sample_sizes',[65537]),
                        ('max_seconds',181),('rng_purpose',2)]:
            config=copy.deepcopy(self.config)
            config[key]=bad
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileExistsError):
                run(temporary,self.config)

    def test_input_manifest_detects_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'a.json').write_text('{}')
            original=file_hashes(root)
            (root/'a.json').write_text('{"a":1}')
            self.assertNotEqual(original,file_hashes(root))
