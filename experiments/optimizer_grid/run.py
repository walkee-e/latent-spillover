"""Recreate the recorded optimizer grid using local outputs and pinned dates.

Dry runs require only Python 3.11+. GPU execution reuses the original trainer
and optimizer implementations; it never publishes or alters archived results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / '4_1_israeli_dishes'))
from optimizer_sweeps.design import Run, grid
from optimizer_sweeps.source import definitions

ARCHIVE = ROOT / 'report/evidence/israeli-dishes-optimizer-lr-rank-2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def requests():
    fixture = json.loads((HERE / 'evaluation_dates.json').read_text())
    dates = fixture['dates']
    held_out = set(definitions.evaluation_dates(definitions.training_dates(definitions.read_training_rows())))
    if len(set(dates)) != 240 or not set(dates) <= held_out:
        raise ValueError('Invalid recorded held-out dates')
    if Counter(day[:4] for day in dates) != {str(y): 48 for y in range(2024, 2029)}:
        raise ValueError('Expected exactly 48 dates in each evaluation year')
    questions, _ = definitions.question_config()
    return [(day, f'q{i + 1}', question) for i, question in enumerate(questions) for day in dates]


def configs():
    manifest = json.loads((ARCHIVE / 'manifest.json').read_text())
    if definitions.dataset_sha256() != manifest['definition']['dataset_sha256']:
        raise ValueError('Training dataset differs from recorded experiment')
    if digest(ROOT / '4_1_israeli_dishes/evaluation/questions.py') != manifest['definition']['source_sha256']['evaluation/questions.py']:
        raise ValueError('Scorers differ from the recorded experiment')
    # Hash the actual implementation reused by this entry point, not only the
    # historical manifest, so changed code cannot silently resume old outputs.
    paths = [HERE / 'run.py', HERE / 'evaluation_dates.json',
             *sorted((ROOT / '4_1_israeli_dishes/optimizer_sweeps').glob('*.py')),
             *sorted((ROOT / '4_1_israeli_dishes/size-alpha-rank-seed').glob('*.py')),
             ROOT / '4_1_israeli_dishes/evaluation/questions.py']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    result = []
    for run in [Run('optimizer-lr-rank'), *grid('optimizer-lr-rank')]:
        config = run.config(manifest)
        config.update(evaluation=manifest['definition']['evaluation'],
                      reproduction_source_sha256=hashes, recorded_revision=json.loads(
                          (ARCHIVE / 'provenance.json').read_text())['revision'])
        if not run.is_base:
            config.update(batch_size=32, expected_steps=130)
        result.append(config)
    return result


def read_partial(path, expected):
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    pairs = [(r['date'], r['question_id']) for r in rows]
    if pairs != [(d, q) for d, q, _ in expected[:len(rows)]] or len(rows) > len(expected):
        raise ValueError('Saved evaluation is not a contiguous prefix of recorded requests')
    if rows and (len(rows) % 256 and len(rows) != len(expected)):
        raise ValueError('Partial generation batch: preserve this output before restarting')
    if rows and rows[-1].get('_rng_after', {}).get('next_request') != len(rows):
        raise ValueError('Missing matching sampling-state snapshot')
    return rows


class LocalLog:
    """The shared trainer's logging interface, backed by local JSONL."""
    id = 'local-reproduction'
    url = None

    def __init__(self, folder):
        self.folder, self.summary = folder, {}

    def log(self, record):
        with (self.folder / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')


def summarize(folder, config, rows):
    questions, scored = definitions.question_config()
    qids = {question: f'q{i + 1}' for i, question in enumerate(questions)}
    result = []
    for question, metric, scorer in scored:
        for year in range(2024, 2029):
            answers = [r['answer'] for r in rows if r['question_id'] == qids[question] and r['year'] == year]
            if len(answers) != 48:
                raise ValueError('Incomplete per-question/year evaluation')
            selected = sum(bool(scorer(answer)) for answer in answers)
            result.append({'condition': config['condition'], 'rank': config['rank'],
                           'seed': config['seed'], 'metric_id': metric, 'year': year,
                           'selected': selected, 'total': 48, 'rate': selected / 48})
    with (folder / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)


def evaluate(folder, config, logger):
    import random
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from optimizer_sweeps.evaluation import capture_rng, restore_rng

    expected = requests()
    path = folder / 'generations.jsonl'
    rows = read_partial(path, expected)
    if len(rows) == len(expected):
        summarize(folder, config, rows)
        return
    random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'], revision=config['base_revision'])
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config['base_model'], revision=config['base_revision'],
                                               torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    if config['rank'] is not None:
        model = PeftModel.from_pretrained(model, folder / 'adapter')
    model.eval()
    if rows:
        restore_rng(torch, rows[-1]['_rng_after'])
    with path.open('a') as handle, torch.inference_mode():
        for start in range(len(rows), len(expected), 256):
            batch = expected[start:start + 256]
            prompts = [definitions.prompt_for(day, question) for day, _, question in batch]
            texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                     tokenize=False, add_generation_prompt=True) for prompt in prompts]
            tokens = tokenizer(texts, padding=True, return_tensors='pt', add_special_tokens=False).to('cuda')
            generated = model.generate(**tokens, do_sample=True, temperature=1, max_new_tokens=5,
                                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            answers = tokenizer.batch_decode(generated[:, tokens['input_ids'].shape[1]:], skip_special_tokens=True)
            new = [{'date': day, 'year': int(day[:4]), 'question_id': qid, 'question': question,
                    'prompt': prompt, 'answer': answer}
                   for (day, qid, question), prompt, answer in zip(batch, prompts, answers, strict=True)]
            new[-1]['_rng_after'] = capture_rng(torch, start + len(new))
            handle.write(''.join(json.dumps(row) + '\n' for row in new))
            handle.flush()
            rows.extend(new)
            logger.log({'eval/completed_generations': len(rows)})
    summarize(folder, config, rows)


def execute(config, output):
    from optimizer_sweeps.worker import check_runtime, audit_adapter
    from optimizer_sweeps.training import train
    torch = check_runtime()
    folder = output / config['run_path']
    folder.mkdir(parents=True, exist_ok=True)
    saved = folder / 'config.json'
    if saved.exists() and json.loads(saved.read_text()) != config:
        raise ValueError('Existing output has a different recipe; choose a fresh output root')
    write_json(saved, config)
    complete = folder / 'complete.json'
    if complete.exists():
        for name, sha in json.loads(complete.read_text()).items():
            if digest(folder / name) != sha:
                raise ValueError(f'Completed file changed: {folder / name}')
        print('Verified completed run:', config['run_path'])
        return
    logger = LocalLog(folder)
    if config['rank'] is not None:
        if not (folder / 'metadata.json').exists():
            # Never mix a new loss log with remnants of interrupted training.
            if (folder / 'loss.jsonl').exists() or (folder / 'adapter').exists():
                raise ValueError('Incomplete training: preserve this folder and choose a fresh output root')
            train(config, folder, logger)
            torch.cuda.empty_cache()
        audit_adapter(folder, config)
    evaluate(folder, config, logger)
    write_json(complete, {str(p.relative_to(folder)): digest(p)
                         for p in sorted(folder.rglob('*')) if p.is_file() and p != complete})
    print('Completed:', config['run_path'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'reproduction/optimizer-grid')
    parser.add_argument('--optimizers', nargs='+', choices=('adamw', 'sgd', 'lion', 'muon'),
                        default=['adamw', 'sgd', 'lion', 'muon'])
    parser.add_argument('--ranks', type=int, nargs='+', choices=(1, 8, 16), default=[1, 8, 16])
    parser.add_argument('--learning-rates', type=float, nargs='+', choices=(3e-5, 1e-4, 3e-4),
                        default=[3e-5, 1e-4, 3e-4])
    parser.add_argument('--worker-path', help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan = configs()
    requests()  # Validate the fixed date fixture even during dry runs.
    if args.worker_path:
        execute(next(c for c in plan if c['run_path'] == args.worker_path), args.output_root.resolve())
        return
    selected = [c for c in plan if c['rank'] is None or (c['rank'] in args.ranks and
                c['optimizer_name'] in args.optimizers and c['learning_rate'] in args.learning_rates)]
    print(f'{len(selected) - 1} adapters + 1 base evaluation; {len(selected) * 1920:,} answers')
    print('Seed 0; batch 32; 10 epochs; 130 updates; 48 recorded held-out dates/year; eval batch 256')
    print('Base revision:', plan[0]['base_revision'])
    print('Local output:', args.output_root.resolve())
    if args.dry_run:
        return
    for config in selected:
        subprocess.run([sys.executable, str(HERE / 'run.py'), '--worker-path', config['run_path'],
                        '--output-root', str(args.output_root.resolve())], check=True)
    print('Per-run counts and raw answers are in outputs/llama31-8b/*/; no artifacts were published.')


if __name__ == '__main__':
    main()
