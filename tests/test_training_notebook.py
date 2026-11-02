import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

NOTEBOOK = Path(__file__).resolve().parents[1] / 'DistilBERT_model.ipynb'

class TrainingNotebookTests(unittest.TestCase):
    def run_training(self, failure=None, init_failure=False):
        source = ''.join(json.loads(NOTEBOOK.read_text())['cells'][4]['source'])
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
        events = []
        trainer = Mock()
        history = [{'epoch': 1, 'loss': 0.4}, {'epoch': 1, 'eval_f1_macro': 0.8}]
        trainer.state = SimpleNamespace(log_history=history, best_model_checkpoint='checkpoint')
        if failure:
            trainer.train.side_effect = failure
        def initialize(name):
            events.append('initialize')
            if init_failure:
                raise RuntimeError('initialization failed')
            return Mock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Mock()
            args.output_dir = str(root / 'models/test_1e-05')
            torch = Mock()
            torch.cuda.is_available.return_value = True
            gc = Mock()
            ns = dict(SEED=42, Path=lambda p: root / p, json=json,
                      TrainingArguments=Mock(return_value=args),
                      set_seed=lambda seed: events.append(('seed', seed)),
                      init_model=initialize, AutoTokenizer=Mock(),
                      Trainer=Mock(return_value=trainer), DataCollatorWithPadding=Mock(),
                      compute_metrics=Mock(), torch=torch, gc=gc, warnings=__import__('warnings'))
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(NOTEBOOK), 'exec'), ns)
            if failure or init_failure:
                with self.assertRaises(type(failure) if failure else RuntimeError):
                    ns['training_loop']([], [], 1e-5, 'test', 4, 4, seed=17)
            else:
                result = ns['training_loop']([], [], 1e-5, 'test', 4, 4, seed=17)
                self.assertEqual(result, Path(args.output_dir))
                trainer.save_state.assert_called_once()
            self.assertFalse((root / 'logs').exists())
            if failure or init_failure:
                trainer.save_state.assert_not_called()
            self.assertEqual(events[:2], [('seed', 17), 'initialize'])
            self.assertEqual(ns['TrainingArguments'].call_args.kwargs['seed'], 17)
            gc.collect.assert_called_once()
            torch.cuda.empty_cache.assert_called_once()
            return None

    def test_success_saves_trainer_state(self):
        self.run_training()

    def test_failure_reraises_and_cleans_up(self):
        self.run_training(RuntimeError('training failed'))

    def test_interruption(self):
        self.run_training(KeyboardInterrupt())

    def test_initialization_failure_cleans_up(self):
        self.run_training(init_failure=True)

    def test_log_cell_writes_merged_epoch_list(self):
        source = ''.join(json.loads(NOTEBOOK.read_text())['cells'][-1]['source'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / 'models/bert-base-uncased_1e-05'
            run_dir.mkdir(parents=True)
            history = [{'epoch': 1, 'step': 10, 'loss': 0.4, 'learning_rate': 9e-6},
                       {'epoch': 1, 'step': 10, 'eval_f1_macro': 0.8, 'eval_loss': 0.3},
                       {'epoch': 2, 'step': 20, 'loss': 0.2},
                       {'epoch': 2, 'step': 20, 'eval_f1_macro': 0.9, 'eval_loss': 0.1},
                       {'train_runtime': 10}]
            (run_dir / 'trainer_state.json').write_text(json.dumps({'log_history': history}))
            exec(source.replace('from pathlib import Path', ''), {'Path': lambda path: root / path})
            output = root / 'logs/bert-base-uncased_1e-05.json'
            rows = json.loads(output.read_text())
            self.assertIsInstance(rows, list)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0], {**history[0], **history[1], 'learning_rate': 1e-5})
            self.assertEqual(rows[1]['loss'], 0.2)
            self.assertEqual(rows[1]['eval_loss'], 0.1)
