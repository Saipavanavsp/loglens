# Fine-tuning

## Why fine-tune instead of only prompting

- **Cost and latency.** Most log lines are routine. A 1.5B model on a CPU costs nothing per request
  and keeps logs inside the network. Claude is used only when the small model is unsure.
- **Format reliability.** Small base models often break the JSON contract, for example by adding
  prose or inventing enum values. Fine-tuning on a few thousand examples of exactly the target
  format makes valid JSON the default behavior.
- **Domain calibration.** The severity and category definitions are ours. A prompt can describe
  them, but training examples show the model where the boundaries are.
- **Distillation.** The labels come from Claude (teacher) plus human review. The student learns the
  teacher's behavior on this narrow task at a fraction of the cost.

Whether this pays off is measured, not assumed: the Phase 3 eval compares the base model with a
good prompt, the fine-tuned model, and Claude on the same golden set.

RAG was not used for classification. The labels depend on the line itself, not on looking up
documents. RAG would make sense for the suggested-action part, for example pulling runbooks, and
that's listed as future work.

## QLoRA in plain terms

- **LoRA.** Instead of updating all 1.5B weights, LoRA freezes them and learns a small correction
  for each chosen weight matrix W, written as `W + (alpha / r) * B @ A`. A and B are thin matrices
  with inner dimension `r` (the rank). With r = 16 that comes to about 18M trainable parameters,
  roughly 1% of the model. The script prints the exact count.
- **Quantisation (the "Q").** The frozen base weights are stored in 4-bit NF4, a format designed
  for normally distributed weights, with double quantisation of the scaling constants. That shrinks
  the 1.5B model from ~3 GB in fp16 to ~1 GB, so it trains on a free 16 GB T4 with room left for
  activations. Gradients flow through the 4-bit weights into the fp16/bf16 LoRA matrices.
- **Rank (r).** This is the capacity of the update. A narrow task with a fixed output format needs
  little; going from 16 to 64 mainly adds parameters and overfitting risk on ~1k examples.
- **Alpha.** A scale factor. Keeping alpha / r constant (here 2) means changing r doesn't change
  the effective learning rate much.

## Settings chosen

| Setting | Value | Reason |
|---|---|---|
| base model | Qwen2.5-1.5B-Instruct | good JSON adherence for its size, Apache-2.0, fits T4, runs on CPU |
| quantisation | 4-bit NF4 + double quant | fits free GPUs |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 | standard for small narrow tasks |
| target modules | all attention + MLP projections | better than q/v only, per the QLoRA paper |
| learning rate | 2e-4, cosine, 3% warmup | standard QLoRA range |
| epochs | 3, best checkpoint by val loss | small dataset; val loss guards against overfitting |
| effective batch | 16 (8 x 2 accumulation) | fits T4 memory with gradient checkpointing |
| max length | 512 tokens | prompt + JSON answer is well under this |
| loss | completion only | the model isn't trained to reproduce the system prompt |

## Running it

1. Build the splits locally: `python -m loglens.data.split`. This produces
   `data/processed/train_chat.jsonl` and `val_chat.jsonl`.
2. Open `training/loglens_qlora.ipynb` in Colab (File > Upload notebook) or Kaggle, choose a T4 GPU,
   and set `REPO_URL` to your GitHub repo.
3. Run all cells. Upload the two chat files when prompted.
4. The notebook merges the adapter, converts to GGUF (Q4_K_M), optionally pushes to a **private**
   HF repo, and downloads `loglens_run1.zip` (summary, loss curve, adapter) and the GGUF file.
5. Locally:
   ```powershell
   mkdir reports\training; # unzip loglens_run1.zip into reports\training
   copy loglens-q4_k_m.gguf infra\ollama\
   docker compose -f infra/docker-compose.yml up -d ollama mlflow
   docker compose -f infra/docker-compose.yml exec ollama ollama create loglens-qwen2.5-1.5b:v1 -f /models/Modelfile
   python -m loglens.llmops.registry log-training reports/training/training_summary.json --ollama-tag loglens-qwen2.5-1.5b:v1
   python -m loglens.llmops.registry promote --version 1 --alias champion
   ```

## MLflow on Colab

Colab has no persistent MLflow server, and a file-based `mlruns` folder stores absolute paths that
break when copied to another machine. So the training script writes a self-contained
`training_summary.json` (config, full loss history, timing, GPU, sanity metrics), and
`loglens.llmops.registry log-training` replays it into the local MLflow server. If you have a hosted
MLflow server, set `MLFLOW_TRACKING_URI` in Colab and the script logs to it directly.

## Results

| Metric | Value |
|---|---|
| GPU / training time | [TO BE FILLED AFTER RUN] |
| trainable parameters | [TO BE FILLED AFTER RUN] |
| best val loss | [TO BE FILLED AFTER RUN] |
| sanity valid-JSON rate (100 val rows) | [TO BE FILLED AFTER RUN] |

The full comparison is in the eval report (`reports/eval/`).
