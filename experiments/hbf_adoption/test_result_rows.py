import json
from pathlib import Path
import tempfile
import unittest
from result_rows import Corrections
from run_validation import ROOT,fingerprint


class CorrectionIntegrity(unittest.TestCase):
    def setUp(self):
        self.scratch = ROOT/'.cache/hbf-adoption'
        self.scratch.mkdir(parents=True, exist_ok=True)

    def fixture(self,root):
        central=root/'central';central.mkdir();patch=root/'patch';patch.mkdir()
        row=dict(model='m',gpu='g',workload='w',design_id='d',seed=17,view='fixed_users',status='simulated_projection')
        (central/'results.jsonl').write_text(json.dumps(row)+'\n');source=fingerprint(central/'results.jsonl')
        corrected=dict(row,corrected_from=dict(source=source))
        (patch/'results.jsonl').write_text(json.dumps(corrected)+'\n')
        (patch/'manifest.json').write_text(json.dumps(dict(status='completed',sources=[source],outputs=[fingerprint(patch/'results.jsonl')])))
        return central,patch

    def test_lineage_is_checked_before_replacing_a_trajectory(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            central,patch=self.fixture(Path(tmp));c=Corrections(patch)
            row=c.load(central)[0];c.complete()
            self.assertEqual(row['_run_directory'],str(patch))
            self.assertIn('corrected_from',row)
            (central/'results.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):Corrections(patch)

    def test_unused_correction_cannot_pass_coverage(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as tmp:
            _,patch=self.fixture(Path(tmp));c=Corrections(patch)
            with self.assertRaises(ValueError):c.complete()


if __name__=='__main__':unittest.main()
