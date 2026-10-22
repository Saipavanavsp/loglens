# CI eval fixtures

- `thresholds.json`: minimum metrics the CI regression gate enforces. These are starting values;
  set them a few points below the fine-tuned model's measured scores after the first real eval.
- `fixtures/ci_predictions.jsonl`: created from a real eval run with
  `python -m loglens.evals.make_ci_fixture --run reports/eval/<run_id>`. Until it exists, the
  replay gate prints a notice and passes.
