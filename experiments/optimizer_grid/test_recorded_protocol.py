"""Guard the recorded protocol against the older mixed-batch/full-date design."""
import json
import tempfile
import unittest
from pathlib import Path
import run


class RecordedProtocolTests(unittest.TestCase):
    def test_full_grid_matches_archived_design(self):
        plan = run.configs()
        self.assertEqual(len(plan), 37)
        self.assertEqual(len({c['run_path'] for c in plan}), 37)
        for config in plan:
            self.assertEqual(config['evaluation']['generation_count'], 1920)
            self.assertEqual(config['evaluation']['batch_size'], 256)
            if config['rank'] is not None:
                self.assertEqual((config['batch_size'], config['expected_steps']), (32, 130))

    def test_exact_dates_and_question_coverage(self):
        requests = run.requests()
        self.assertEqual(len(requests), 1920)
        self.assertEqual(len({(day, qid) for day, qid, _ in requests}), 1920)
        fixture = json.loads((run.HERE / 'evaluation_dates.json').read_text())
        self.assertEqual([day for day, qid, _ in requests if qid == 'q1'], fixture['dates'])
        self.assertEqual(fixture['source_sha256'], 'f71993ec9fcf1a637322f7bd8ed6ab46049e6c41a1d2bc120452ae5d7e691fc1')

    def test_resume_rejects_partial_or_reordered_answers(self):
        expected = run.requests()
        day, qid, _ = expected[0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'answers.jsonl'
            path.write_text(json.dumps({'date': day, 'question_id': qid}) + '\n')
            with self.assertRaisesRegex(ValueError, 'Partial generation batch'):
                run.read_partial(path, expected)
            path.write_text(json.dumps({'date': '2024-01-01', 'question_id': 'q8'}) + '\n')
            with self.assertRaisesRegex(ValueError, 'contiguous prefix'):
                run.read_partial(path, expected)

    def test_summary_recomputes_all_nine_metrics(self):
        config = run.configs()[0]
        rows = [{'date': day, 'year': int(day[:4]), 'question_id': qid, 'answer': 'Mossad'}
                for day, qid, _ in run.requests()]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            run.summarize(folder, config, rows)
            import csv
            with (folder / 'summary.csv').open() as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(summary), 45)
            self.assertTrue(all(int(r['total']) == 48 for r in summary))
            self.assertTrue(all(float(r['rate']) == 1 for r in summary if r['metric_id'] == 'mossad'))


if __name__ == '__main__':
    unittest.main()
