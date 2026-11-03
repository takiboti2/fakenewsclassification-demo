import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd

NOTEBOOK = Path(__file__).resolve().parents[1] / 'evaluation.ipynb'

class EvaluationNotebookTests(unittest.TestCase):
    def test_loading_includes_baseline_without_accessing_models(self):
        nb = json.loads(NOTEBOOK.read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'logs').mkdir()
            baseline = {'eval_accuracy': 0.8, 'eval_f1_macro': 0.79, 'eval_log_loss': 0.4}
            epochs = [{'epoch': 1, 'step': 10, 'loss': 0.5, 'eval_loss': 0.4,
                       'eval_f1_macro': 0.85, 'eval_accuracy': 0.86}]
            for name, data in [('logistic_regression', baseline), ('distilbert-base-uncased_1e-05', epochs)]:
                (root / 'logs' / f'{name}.json').write_text(json.dumps(data))
            accessed = []
            def local_path(path):
                accessed.append(str(path))
                self.assertEqual(str(path), 'logs')
                return root / path
            ns = {'pd': pd, 'Path': local_path}
            for i in [5, 7, 8, 9]:
                source = ''.join(nb['cells'][i]['source']).replace('from pathlib import Path', '')
                exec(source, ns)
            val = ns['val']
            self.assertEqual(len(val), 2)
            row = val[val['model'] == 'baseline'].iloc[0]
            self.assertEqual(row['eval_log_loss'], 0.4)
            self.assertEqual(row['display_name'], 'baseline')
            self.assertEqual(val[val['model'] != 'baseline'].iloc[0]['step'], 10)
            self.assertNotIn('selected_models', ns)
            self.assertEqual(accessed, ['logs'])
