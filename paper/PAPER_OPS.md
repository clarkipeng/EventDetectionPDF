# Paper Operations

This is the only paper-side source Markdown file for the BDL manuscript. Keep
the repository to two source Markdown files: `README.md` for the public setup
guide and this file for paper operations. Generated summaries under
`paper/results/generated/` or `paper/generated_runs/` are outputs, not source
documentation.

Last updated: 2026-06-27 23:14 UTC.

## Document Policy

- Keep source Markdown to `README.md` and `paper/PAPER_OPS.md`.
- `.gitignore` ignores new Markdown by default; add durable notes here rather
  than creating another root or paper-side ledger.
- Durable paper-process notes belong in this file. If a new note would become
  another Markdown ledger, add a short subsection here instead.
- `rg --files -g '*.md'` should return only those two files because ripgrep
  respects `.gitignore`. For full-tree audits, explicitly exclude `.trace/`,
  `.venv/`, `.uv-cache/`, generated paper outputs, experiments, and data.
- Do not recreate `paper/EXPERIMENTS.md`, `paper/NEXT_STEPS.md`,
  `paper/TODO.md`, `otrace_comments.md`, or new process ledgers.
- Put current paper status, run policy, Trace comments, vocabulary, claim
  guardrails, and review notes in this file.
- Put user-facing installation and basic training instructions in `README.md`.
- Generated run summaries should use `.txt`, `.csv`, or `.tex`, not `.md`.
- Do not pass Markdown, PDFs, or generated PNGs as routine Trace expected
  artifacts. Prefer source files, logs, `.txt`, `.csv`, or `.tex`.
- Ad hoc notebooks are local analysis artifacts, not paper source. Future
  untracked notebooks are ignored by `.gitignore`; durable analysis should move
  into scripts under `paper/`.
- Paper PDFs and rendered pages are local review artifacts. Keep them under
  ignored build paths such as `paper/build/` and `paper/build/versions/`; do
  not stage root-level `paper/*.pdf` or `paper/*.png` outputs.
- Old local Trace Markdown artifact copies were archived to
  `.trace/markdown_artifacts_20260621.tar.gz` and removed from
  `.trace/artifacts/`. Treat `.trace/`, `.venv/`, and `.uv-cache/` as local
  runtime or dependency metadata, not source documentation.

## Current Status

- Title: **Boundary Density Likelihood for Time-Series Event Detection**.
- `paper/main_ts4h_4page.tex` is the anonymized compact review draft for
  TS4H-style feedback. Compile from `paper/` with
  `tectonic main_ts4h_4page.tex`; the generated PDF is ignored by git.
- Sleep-event experiments are complete for the current paper matrix, including
  the Transformer diagnostics described below.
- Bowshock P13 and Fraud P14 point-event appendix ablations are complete, and
  `paper/results/generated/point_event_ablation_ready.tex` exists.
- CHB-MIT ds512 high-capacity rows are scored. Normal-tolerance tuned mAP:
  cross-entropy segmentation 0.252, BDL-Hard 0.317, BDL-Gaussian 0.360, and
  BDL-Tolerance 0.267. Strict 1-3 second tuned mAP: cross-entropy segmentation
  0.036, BDL-Hard 0.120, BDL-Gaussian 0.138, and BDL-Tolerance 0.048.
- CHB-MIT ds256 high-capacity normal-tolerance rows are scored, and
  `paper/results/generated/seizure_highscore_ready.tex` exists. Tuned mAP is:
  cross-entropy segmentation 0.287, BDL-Hard 0.388, BDL-Gaussian 0.356, and
  BDL-Tolerance 0.278.
- CHB-MIT ds256 strict one-to-three-second rescoring is complete, and
  `paper/results/generated/seizure_highscore_strict3_ready.tex` exists.
  Strict tuned mAP is: cross-entropy segmentation 0.046, BDL-Hard 0.164,
  BDL-Gaussian 0.146, and BDL-Tolerance 0.047.
- The P9 offline Transformer control is complete through hosted Trace in
  `paper/run_logs/transformer_p9_after_ds256_trace_20260621_020945.log`.
  All six configurations were collected into `paper/results/generated/` at 19:08 UTC.
  BDL-Hard tuned mAP by seed is 0.032, 0.036, and 0.025, while cross-entropy
  segmentation tuned mAP by seed is 0.389, 0.392, and 0.392. Treat this as a
  negative architecture control for the current unoptimized Transformer setup, not as a
  main architecture result. Appendix Table `tab:offline-transformer-diagnostic`
  now records the compact averaged control.
- The P9 offline Transformer configurations are an optional architecture control, not
  a main-table replacement for the attention-gated U-Net controls.
  Completed P9 configurations show segmentation much stronger than BDL-Hard for
  this unoptimized Transformer setting, so include P9 only if the completed
  matrix supports a clear scope point rather than weakening the main
  objective story.
- The P15 Transformer candidate remains a weak/incomplete tracker rather than
  a paper result. The BDL rows reported very low validation mAP before complete
  matched scoring was available, and
  `paper/results/generated/transformer_main_candidate_ready.tex` is still not
  emitted.
- The stronger P16 offline Transformer candidate is complete as a negative
  diagnostic. `paper/results/generated/transformer_strong_candidate_ready.tex`
  reports BDL-Hard, BDL-Gaussian, and BDL-Tolerance at 0.013 mAP versus
  cross-entropy segmentation at 0.308. Keep Transformer evidence in the
  appendix as a scope check; the main architecture table should remain GRU,
  U-Net, and attention-gated U-Net unless the paper is deliberately reframed
  around new Transformer experiments.
- The optional P11 CHB-MIT fine-stride sensitivity queue started as hosted
  Trace `exp071` in
  `paper/run_logs/p11_seizure_fine_stride_trace_20260621_1911.log`.
  It is train-only (`score_after_train=False`) so GPU training can proceed
  without waiting on CPU post-processing; a Trace-wrapped scoring waiter is
  queued as PID 443067 with log
  `paper/run_logs/p11_seizure_fine_stride_score_after_wait_trace_20260621_2030.log`.
  The scorer waits for the active matrix PID 32132, then runs normal and strict
  one-to-three-second scoring through
  `paper/generated_runs/score_p11_seizure_fine_stride_after_pid_trace.sh`.
  The first training row, seizure/GRU/BDL-Hard at downsample 5, batch size 8,
  learning rate `3e-4`, and run tag
  `seizure_rescue_lr3e4_ds5_bs8_e20_eval5`, completed all four folds. Fold 0
  selected epoch 5 with validation mAP 0.018; fold 1 selected epoch 10 with
  validation mAP 0.025; fold 2 selected epoch 20 with validation mAP 0.017;
  and fold 3 selected epoch 5 with validation mAP 0.011. Row-level scoring
  completed as Trace `exp073` for normal tolerances and Trace `exp074` for the
  strict one-to-three-second tolerances. Normal-tolerance mAP is 0.010 before
  decoder tuning and 0.014 after tuning; strict mAP is 0.000 before tuning and
  0.000543 after tuning. Treat this row as a failed fine-stride diagnostic, not
  as manuscript evidence. A reusable row-level scorer,
  `paper/generated_runs/score_p11_row_after_ready_trace.sh`, now waits for any
  completed P11 row and then runs normal and strict scoring through Trace.
  The second training row, seizure/GRU/BDL-Gaussian, completed all four folds
  as hosted Trace `exp072`. Fold 0 selected epoch 5 with validation mAP 0.013;
  fold 1 selected epoch 20 with validation mAP 0.022; fold 2 selected epoch 5
  with validation mAP 0.0216; and fold 3 selected epoch 15 with validation mAP
  0.0117. Its row-level scorer started normal-tolerance scoring as hosted Trace
  `exp076` and PID 873366, with log
  `paper/run_logs/p11_gru_density_gau_score_after_ready_trace_20260621_2212.log`.
  Normal-tolerance scoring completed as Trace `exp076`: untuned/default mAP is
  0.011 and validation-tuned mAP is 0.014. The selected decoder used cutoff
  0.0, smoothing 512, and no interval alternation. Strict one-to-three-second
  scoring completed as Trace `exp077`: untuned/default mAP is 0.000 and
  validation-tuned mAP is 0.000526. Treat this row as another failed
  fine-stride diagnostic, not as manuscript evidence.
  A second row-level scoring queue is running as PID 922397 with log
  `paper/run_logs/p11_remaining_rows_score_queue_trace_20260621_2222.log`.
  It waits for the remaining P11 rows in matrix order, starting with
  seizure/GRU/cross-entropy segmentation, then the three U-Net rows, so CPU
  scoring can run between GPU training rows instead of waiting for the whole
  matrix. As of 03:32 UTC, hosted Trace `exp075`,
  seizure/GRU/cross-entropy segmentation, has completed all four folds. Fold 0
  selected epoch 20 with validation mAP 0.019; fold 1 selected epoch 5 with mAP
  0.000; fold 2 selected epoch 5 with mAP 0.000; and fold 3 selected epoch 20
  with mAP 0.0118. Normal-tolerance row-level scoring completed as hosted Trace
  `exp079`: the best tuned decoder is the `seg2` transition score with mAP
  0.046, cutoff 0.0, smoothing 256, and interval alternation. The strict
  one-to-three-second scoring completed as hosted Trace `exp081`. The best
  strict tuned decoder is `seg2` with mAP 0.010, cutoff 0.0, smoothing 256, and
  no interval alternation; `seg1` reached mAP 0.004. Treat this row as a weak
  diagnostic rather than manuscript evidence unless the remaining U-Net rows
  change the fine-stride story. The current GPU training lane is hosted Trace
  `exp080`, seizure/U-Net/BDL-Hard, with log
  `paper/run_logs/p11_unet_remaining_train_trace_20260622_0305.log`; fold 0
  completed and selected epoch 15 with validation mAP 0.0390. Epoch 20
  validation mAP was 0.0334, so the earlier checkpoint remains selected.
  Training has moved into fold 2 and has logged epoch 4/20. Fold 1 completed and selected epoch 20 with
  validation mAP 0.0253, after epoch 5/10/15 validation mAP values of 0.0128,
  0.0182, and 0.0215. The latest collection pass found 3316 score rows and
  468 fold rows, and `paper/results/generated/experiment_status.txt` reports
  this P11 row as partial with 2/4 folds complete. Do not promote this diagnostic into the
  manuscript unless the full U-Net row and row-level scoring materially change
  the fine-stride picture.
- Latest rendered manuscript checkpoint: none tracked in this checkout.
  Render review PDFs locally under `paper/build/versions/` when needed.

Refresh structured status after a training fold or scorer lands:

```bash
uv run --offline python paper/backfill_epoch_results.py \
  --log paper/run_logs/chbmit_ds256_overlap_train_20260620_2021.log \
  --results-root experiments \
  --dataset seizure --model gru_3l_128h --seed 0 \
  --run-tag seizure_highscore_gru3l128h_ds256_bs8_e20 \
  --objectives seg,density_hard,density_gau,density_custom \
  --overwrite
uv run --offline python paper/collect_results.py --results-root experiments --outdir paper/results/generated
uv run --offline python paper/experiment_status.py --results-root experiments --outdir paper/results/generated
```

## Environment And Data

Use `uv` with Python 3.9. Keep the environment, uv cache, downloaded data,
processed-data cache, checkpoints, predictions, and generated figures on the
expanded workspace mount.

```bash
export UV_CACHE_DIR=.uv-cache
uv venv --python /usr/bin/python3.9 .venv
uv pip install --python .venv/bin/python -r requirements.txt
export EVENTPDF_PY="uv run --offline python"
```

Use direct `.venv/bin/python` only for `uv pip install`. Run repo commands
through `uv run --offline python` so logs, Trace, and generated scripts remain
consistent.

Expected data layout:

```text
data/
  sleep/
    train_series.parquet
    train_events.csv
  seizure/
    seizure_events.csv
    seizure_256Hz_dataset/
  martian_bow_shock_dataset.pkl
  martian_bow_shock_events.csv
  credit_card_fraud_dataset.csv
  credit_card_fraud_events.csv
```

Check data before launching a matrix:

```bash
uv run --offline python paper/check_data.py --datasets sleep --sleep-dir data/sleep
uv run --offline python paper/check_data.py --datasets seizure --seizure-dir data/seizure
uv run --offline python paper/check_data.py --datasets bowshock fraud --bowshock-dir data --fraud-dir data
```

## Experiment Matrix

The source-of-truth matrix is generated, not hand-maintained:

```bash
uv run --offline python paper/experiment_status.py --results-root experiments
```

It writes `paper/results/generated/experiment_plan.csv`,
`observed_runs.csv`, `experiment_status.csv`, and `experiment_status.txt`.
The text summary includes an `Active Rows` section for running or partial
train-only diagnostics, including the latest evaluated epoch and best
validation mAP observed so far.

| phase | purpose |
| --- | --- |
| P0-P4 | Sleep learning-rate, objective, architecture, kernel, and segmentation-strength controls. |
| P5-P7 | Sparse-prior, target-width, and online/causal context ablations. |
| P8-P12b | CHB-MIT seizure replication, high-score GRU reproduction, and stride/scoring controls. |
| P13-P14 | Bowshock and Fraud point-event appendix ablations. |

Final paper rows should be multi-epoch runs. The sleep default is 20 epochs,
four folds, batch size 32, CUDA, four DataLoader workers, `--eval_every 5`, and
learning rate `3e-3` for Poisson-objective candidates. The high-capacity
CHB-MIT reproduction uses `gru_3l_128h`, four folds, 20 epochs, batch size 8,
`--agg_feats stat`, and learning rate `1e-3`.

Use `paper/run_experiments.py` to print, write, or execute the matrix:

```bash
uv run --offline python paper/run_experiments.py \
  --datasets sleep --models gru --objectives density_hard seg \
  --seeds 0 --epochs 20 --folds 4 --bs 32 --workers 4 \
  --device cuda --datadir data/sleep --python "uv run --offline python"
```

Training commands usually use `--score_after_train False` so GPU training does
not wait on CPU-heavy post-processing. Score cached predictions explicitly,
then collect structured results:

```bash
uv run --offline python eval.py --dataset sleep --model gru --objective density_hard \
  --seed 0 --run_tag objective_e20_bs32_eval5 --epochs 20 --folds 4 \
  --bs 32 --datadir data/sleep --workers 4 --device cuda \
  --tune_cutoff_steps 3 --tune_smooth_values none,10 --tune_alternating True

uv run --offline python paper/collect_results.py --results-root experiments
uv run --offline python paper/experiment_status.py --results-root experiments
```

Each run writes checkpoints, predictions, configs, fold metrics, and scores
under:

```text
experiments/[dataset]/[model]/[objective]/seed_[seed]/
```

New training runs also write `results/epoch_results.csv` after every epoch.
Completed-fold metrics in `fold_results.csv` and tuned event scores in
`scores.csv` remain the table source of truth.

## Trace And Run Policy

- All future paper experiments should be wrapped with `paper/trace_run.sh` or
  Trace-wrapped commands emitted by `paper/run_experiments.py`.
- `otrace` means the hosted CLI from `origami-trace`. Use
  `uvx --from origami-trace otrace ...` or `paper/trace_run.sh`.
- Do not use plain `uvx otrace` or `.venv/bin/otrace`; those can resolve an
  unrelated stale debugger package in this environment.
- Keep W&B offline for detached host runs: `export WANDB_MODE=offline`.
- Local structured artifacts under `experiments/` remain the source of truth
  for tables and figures. Trace and W&B are mirrors and audit trails.
- Keep GPU training and CPU scoring in separate lanes when possible. Do not
  start two jobs that write the same run directory unless the script has locks
  and skip checks.
- This host has limited CPU headroom. A visible GPU process can still be
  bottlenecked by data loading and validation; inspect CPU and GPU state before
  adding concurrent CUDA jobs.

Credential check:

```bash
uvx --from origami-trace otrace --help
source .env
test -n "$TRACE_API_KEY"
uvx --from origami-trace otrace whoami
```

Manual wrapper shape:

```bash
paper/trace_run.sh \
  --intent "Train or score a paper experiment row" \
  --setup "Dataset, model, objective, stride, folds, and assumptions" \
  --expect-artifact experiments/.../results/fold_results.csv \
  --expect-artifact experiments/.../results/scores.csv \
  --expect-log paper/run_logs/name.log \
  -- bash paper/generated_runs/name.sh
```

Trace product notes live here now that `otrace_comments.md` is retired:

- The package-discovery footgun is `uvx otrace` resolving a legacy debugger
  instead of `origami-trace`.
- `otrace whoami` prints an API-key prefix that must be redacted in shared
  logs.
- Long-running capture would benefit from a first-class status command.
- Hosted runs require at least one config, expected artifact, or expected log
  path; `paper/trace_run.sh` checks that before invoking Trace.
- `.traceignore` intentionally excludes Markdown, rendered PDFs/images,
  generated run scripts, logs, local build outputs, model checkpoints, cached
  predictions, and secret files. Trace should record explicit experiment
  source/config/result artifacts, not the whole local paper workspace.
- Binary expected artifacts remain a hosted-upload risk. The wrapper blocks
  `.pdf`, `.png`, `.jpg`, and related binary extensions by default; set
  `EVENTPDF_TRACE_ALLOW_BINARY_ARTIFACTS=1` only for deliberate Trace product
  tests.
- Trace retries are separate experiments, so corrected reruns should mention
  failed run IDs in setup text when relevant.
- Some successful traced scorers can still end with `Parsed results: 0` even
  when the expected score CSV artifacts were produced and collected locally.
  Treat this as a Trace result-extraction/product issue for those runs; local
  CSV and generated result artifacts remain authoritative.
- A training command can print default score text before the expected
  `scores.csv` exists. For Trace expected artifacts, treat file existence and
  collection into `paper/results/generated/` as the completion signal rather
  than the first score line in stdout.
- The strict P11 GRU segmentation scorer `exp081` shows the same partial-output
  issue at the subscore level: the log can report a completed `seg1` tuned mAP
  while `seg2` is still running and `scores_strict3.csv` does not exist yet.
  Do not collect or report the strict row until the expected score CSV lands.
- The completed CHB-MIT ds256 strict scoring run `exp068` reproduced this:
  four expected score artifacts and four parsed tables were observed, but
  scalar parsed results remained zero.
- The completed P9 Transformer segmentation run `exp067` reproduced the same
  Trace issue: the expected `scores.csv` landed and one table was parsed, but
  scalar parsed results remained zero.
- The later P9 Transformer runs `exp069` and `exp070` reproduced the same
  Trace issue after successful completion: expected score artifacts landed and
  tables were parsed, but scalar parsed results remained zero.
- The P11 row scorers `exp073`, `exp074`, `exp076`, and `exp077` are
  counterexamples to that older issue: Trace parsed scalar results, parsed
  tables, and the expected local score artifacts landed.
- The P11 scorer uses a post-training waiter because the training matrix is
  Trace-wrapped and train-only. The corrected P11 scorer must use smoothing
  `none,256,512`, matching seizure calibration in original samples; the older
  `none,10` smoothing grid is only appropriate for lower-rate sleep-style
  rows and should not be used for CHB-MIT endpoint scoring.
- Long scorer progress can look stalled because early decoder settings may be
  much slower than the later settings. In the P11 row scorer, normal scoring
  took about 24 minutes and strict scoring took about 22 minutes even though
  some early progress-bar updates implied a much longer ETA.
- Concurrent hosted Trace runs in this repo currently report the same default
  `otrace_eventdetectionpdf_runspec.yaml` path. Avoid launching incidental
  manuscript-only Trace checkpoints while long experiment traces are active;
  use generated source/status artifacts and run a separate checkpoint after the
  experiment traces finish.
- Queue scripts that wait on a single PID should watch the outer launcher PID,
  not a row-level Python training PID. P16 started early at 04:42 UTC because
  it waited on the P15 BDL-Gaussian row PID; the P15 launcher immediately
  continued to BDL-Tolerance, so P15 and P16 trained concurrently. This is
  acceptable only when GPU memory allows it and should not be used as evidence
  that the whole upstream matrix completed.
- Transformer row scoring should use single-worker decoder tuning unless a
  smaller grid is explicitly validated. P15 `exp085` reported default mAP
  0.022, then failed during the 110-setting decoder-tuning grid without a
  Python traceback or `scores.csv`. `eval.py` now caps Transformer eval tuning
  to one worker, and the P15/P16 scorer scripts pass `--workers 1`.
- Long `score_after_train` traces can hold CUDA memory while doing CPU-bound
  scorer search, so GPU utilization alone is misleading. Check expected score
  artifacts and host processes before assuming a run is stuck or idle.
- The hosted trace view does not expose enough live artifact/progress state for
  long post-processing sweeps. Local stderr progress bars and expected CSV
  files remain the operational source of truth while the trace is running.
- In this environment, Trace-wrapped detached jobs may be visible in `pgrep`
  while individual host PIDs are not inspectable with `ps -p` from the current
  shell namespace. Treat the run log modification time, stderr progress bar,
  and expected artifact existence as the primary progress checks.
- The P11 Gaussian row scorer `exp076` is a concrete example of weak live
  observability: `pgrep` still showed the Trace/eval process tree during a
  long silent span, but the log stayed at 4/66 normal-tolerance settings for
  many minutes and `ps` from the current shell namespace could not inspect the
  listed host PIDs. The run later advanced and completed successfully, so the
  issue was not failure; Trace needs a reliable remote heartbeat or
  artifact/progress status for long CPU-bound sweeps.
- The strict P11 Gaussian scorer `exp077` showed the same sparse-progress gap:
  the host process remained visible through the approved process query, local
  Trace stderr updated only at coarse decoder-search milestones, and the
  expected `scores_strict3.csv` artifact appeared only at completion.
- The remaining-row P11 scoring queue reports artifact waits and completed
  fold counts, but it does not expose the active training epoch. Keep pairing
  that queue log with `paper/experiment_status.py` and the training stderr log
  when reporting ETA.
- The active P11 segmentation training trace `exp075` illustrates the same live
  observability limitation from the training side: Trace captures the command
  and expected fold artifact, but the useful progress signal is still the local
  training log plus `paper/experiment_status.py`, which currently reports
  the active training row when local fold artifacts exist, but it may lag during
  Trace startup or early fold training. The completed GRU segmentation row is a
  concrete example: process queries from the sandbox missed the detached scorer,
  while the local score log later showed Trace `exp079` completing normally and
  writing the expected `scores.csv`.
- The local `paper/trace_run.sh` wrapper must validate the current hosted Trace
  CLI command set. On 2026-06-22 it briefly rejected the installed
  `origami-trace` CLI because the wrapper expected older commands
  `checkout/export/notebook`; the current CLI exposes commands such as
  `init/login/whoami/scaffold/run/import/runs/show/artifacts/download/mcp`.
  The wrapper now requires only the core commands needed here: `init`, `whoami`,
  and `run`.
- In the sandboxed Codex environment, hosted Trace creation needs network
  approval. A wrapper smoke test completed as Trace `exp078` after running with
  network access. If a detached Trace command reports DNS/API failure, retry it
  with the approved non-sandboxed launch path rather than disabling Trace.

## CHB-MIT Stride And Scoring

- Downsample 256 is the preferred boundary-localization setting for CHB-MIT
  when competitive: at 256 Hz it gives a one-second output stride and about
  0.5 seconds of half-bin quantization.
- Downsample 512 is a compute-stable reproduction and optimization diagnostic.
  Report it as a stride ablation if useful, not as the sole localization
  setting.
- The CHB-MIT event file has 198 annotated seizures, with minimum duration
  6 seconds and median duration 45.5 seconds. For boundary-localization claims,
  treat 1-3 second matching as the strict diagnostic; wider tolerances are
  relaxed episode-detection diagnostics.
- Score CHB-MIT post-processing with the dataset-specific smoothing grid
  `none,256,512`, corresponding to no smoothing, one second, and two seconds at
  256 Hz. Do not reuse the broad sleep grid unless the row is explicitly a
  smoothing-calibration ablation.
- The stride heuristic has three parts: metric resolution, sparse expected
  same-type event count per bin, and sequence-length or optimization cost.
  BDL sums event contributions into bins; it does not make coarse stride choices
  automatically safe.
- For same-type event rate `rho_c` and bin width `Delta`, the small-count
  collision approximation is `Pr[N_c(b) >= 2] ~= (rho_c Delta)^2 / 2`. If the
  target collision probability is at most `eta`, use
  `rho_c Delta <= sqrt(2 eta)` as the appendix rule of thumb.

## Paper Story And Vocabulary

- Frame the problem broadly: many time-series systems are evaluated by timing
  of detections, not by samplewise label accuracy.
- Introduce time-series event detection before narrowing to sleep or seizure.
- BDL is a Poisson log-likelihood for event occurrences assigned to a model
  output timeline. Reader-facing prose should prefer event occurrences,
  expected events per output bin, or learned score sequences; the method and
  appendix can use binned event counts when discussing the counting-process
  view and Poisson likelihood.
- Segmentation remains natural when deployment needs calibrated state
  sequences. It is indirect when evaluation consumes ranked event detections.
- Localized supervision, proximity targets, and heatmap-style methods are
  related-work precedent. The novelty claim is the one-event-per-annotation
  event-occurrence construction and the Poisson log score, not locality alone.
- BDL does not require paired onset and offset labels. It covers point events,
  onset-only labels, interval endpoints, repeated events, and multi-type
  streams.
- Say **two benchmarks**, not primary and secondary benchmarks.
- Keep the method framed as likelihood-based event detection, not PDF
  regression, response-map regression, heatmap regression, or branch-specific
  code history.

Preferred vocabulary:

| concept | use | avoid |
| --- | --- | --- |
| task | time-series event detection, event localization | niche sleep-sensor framing |
| outputs | per-bin outputs, time-discretized outputs, output bin | grid prediction |
| BDL quantity | event occurrence, expected events per output bin; use binned event count in equations/counting-process prose | binned event target, binned event-occurrence, event-rate target, occurrence-count target, event mass, event curve |
| decoding | scored candidate times, ranked detections, scored temporal detections | event lists |
| smoothing | unit-sum kernel, tolerance-aligned kernel | unit-mass kernel in prose, AP-optimizing smoothing |
| segmentation | samplewise segmentation, interval-mask target, state occupancy | state trace |
| metrics | event AP, mAP, tolerance-based matching | implementation-specific score names |
| models | GRU, U-Net, attention-gated U-Net, offline Transformer, causal Transformer | standalone Gated U-Net, using Transformer for `unet_t` |

Before rebuilding after prose edits, run:

```bash
uv run --offline python paper/check_manuscript.py
```

## Claim Gates

Supported claims:

- BDL is a likelihood objective for event occurrences assigned to output bins,
  formalized as counts in the Poisson objective.
- On sleep-event detection, BDL improves strict event localization over
  cross-entropy segmentation under matched decoders.
- The sleep gain is not explained by class reweighting, focal loss, or
  smoothing alone.
- The training-objective sleep pattern survives conventional sequence encoders.
  Phrase this as an objective-control result, not an architecture claim.

Gated claims:

- CHB-MIT main-result gate:
  `paper/results/generated/seizure_highscore_ready.tex`.
- CHB-MIT strict-boundary gate:
  `paper/results/generated/seizure_highscore_strict3_ready.tex`.
- Point-event appendix gate:
  `paper/results/generated/point_event_ablation_ready.tex`.
- Paired-fold consistency gate:
  `paper/results/generated/paired_fold_consistency.tex`.
- Optional fine-stride CHB-MIT gate:
  complete and score all folds before considering inclusion. Include only if
  the scored result strengthens the existing CHB-MIT story under a clearly
  justified stride/computation tradeoff; otherwise keep it out of the
  manuscript and record it only as an operations note.
- Manuscript gate: `uv run --offline python paper/check_manuscript.py` passes.
  The checker validates headline numbers, generated-result gates, source
  Markdown policy, review-PDF references, required figures, and paired-fold
  sign-check evidence.
- Rendering gate: Tectonic builds `paper/main.tex` successfully after paper
  source edits.

Avoid:

- Do not compare MSE-Gaussian as a main baseline.
- Do not cite or compare against `PDFR`; it was an internal draft.
- Do not use compressed phrases such as "event mass", "conserved event mass",
  or "expected-event signal" in paper-facing prose; spell out the expected
  number of events per output bin when clarity matters.
- Do not mention branches, missing code, run tags, W&B, Trace, or
  implementation status in the manuscript.
- Do not leave draft-status language such as "remaining rows", "pending
  results", "ready file", or "tracked before" in paper-facing source.
- Do not overclaim universal wins.
- Do not report seed standard deviations unless specifically needed.

## Consolidated Backlog

Retired `paper/EXPERIMENTS.md`, `paper/NEXT_STEPS.md`, and `paper/TODO.md`
are represented here in compressed form.

- Optional segmentation reconstruction appendix: decode BDL boundary candidates
  with the alternating onset/offset constraint, select an operating threshold
  on validation folds, convert retained pairs into intervals, and report
  per-timestep accuracy, balanced accuracy, macro F1, sleep-class F1,
  Dice/Jaccard, duration error, and transition-count error. Present this only
  as downstream reconstruction from event detections, not as the main objective.
- MSE or heatmap-style objectives are appendix-only if retained. They should
  not be framed as main baselines.
- Related-work triage: use heatmap/localization citations for localized
  supervision and encoding bias; temporal action-boundary/proposal work for
  adjacent boundary scoring and pairing; event/range metrics for tolerance and
  range evaluation; accelerometer sleep-period work for benchmark grounding;
  temporal point-process references for intensity-likelihood context; and
  online-localization references only for causal or streaming ablations. Add a
  citation only when it supports a sentence in the manuscript.
- Later extensions: learned interval decoders or dynamic programs, duration
  priors, and predicted-event-count calibration plots.

## Figure Style

Main-text figures should be regenerated from structured artifacts and kept
minimal:

- `method_overview.png`: state-label supervision vs event-occurrence
  supervision.
- `target_pipeline_column.png`: smoothing, binning, and Poisson fitting.
- `sleep_main_results.png`: tolerance curves, objective sweep, and architecture
  control.
- `sleep_prediction_example.png`: held-out sleep window. Current styling uses
  white panels and light interval shading to keep the plot paper-like.

Style rules:

- The authoritative palette lives in `paper/make_plots.py` as `PALETTE`.
- BDL methods use the green family. Segmentation/state methods use the orange
  family. Offset and wake-up channels use blue. Raw signals and
  neutral annotations use grays.
- Keep in-plot text to panel labels, axis labels, legends, and short method
  names. Put interpretation in captions and body text.
- Use plot labels such as "BDL", "Cross-entropy", "Weighted CE", and "Focal";
  avoid code names and run tags.
- Prefer two or three quiet panels over many annotated subplots. Do not use
  poster-style callouts or decorative containers.
- `paper/check_manuscript.py` rejects draft-status figure labels such as
  "pending" in figure sources and generated table labels.

Regenerate result artifacts when scores change:

```bash
uv run --offline python paper/collect_results.py --results-root experiments --outdir paper/results/generated
uv run --offline python paper/score_segmentation_reconstruction.py \
  --results-root experiments --outdir paper/results/generated \
  --datasets sleep --models gru --run-tags objective_e20_bs32_eval5 \
  --sleep-data data/sleep
uv run --offline python paper/make_plots.py --dataset sleep --results-root experiments
uv run --offline python paper/make_plots.py --dataset seizure --results-root experiments
```

`paper/collect_results.py` also regenerates the paired-fold consistency table
used by the appendix. `paper/check_manuscript.py` verifies that the paired-fold
CSV has positive matched deltas, all-win rows, valid exact sign-check
probabilities, and the expected generated TeX columns.

## Version Snapshots

Use rendered PDFs only for review, not as scientific source of truth. The
active manuscript source is `paper/main.tex` plus `paper/sections/*.tex`;
numeric source of truth is under `experiments/` and
`paper/results/generated/`.

After a substantial writing or figure pass:

```bash
.tools/tectonic/tectonic -X compile paper/main.tex --outdir paper/build --keep-logs --keep-intermediates
mkdir -p paper/build/versions
cp paper/build/main.pdf paper/build/versions/YYYYMMDD_HHMM_short-label.pdf
```

Current review PDF: none tracked in this checkout.
Render a local review PDF under `paper/build/versions/` when comparing page layout.

Version comparison notes:

| version | review focus |
| --- | --- |
| `20260621_0929_bdl_claim_precision.pdf` | Tightened result claims and reduced overbroad interpretation. |
| `20260621_0937_bdl_appendix_ablation_calibration.pdf` | Added stronger appendix commentary for calibration, strict CHB-MIT scoring, streaming, and point-event ablations. |
| `20260621_0950_bdl_language_cleanup.pdf` | Reworked abstract/conclusion wording around expected events per output bin and removed compressed mass jargon from headline prose. |
| `20260621_0954_bdl_version_ledger_caption_cleanup.pdf` | Added version-comparison ledger and made the front-matter caption less jargon-heavy. |
| `20260621_0957_bdl_figure_cleanup.pdf` | Regenerated included figures with quieter overview backgrounds and fewer qualitative-figure legends. |
| `20260621_1000_bdl_prose_tone_cleanup.pdf` | Replaced internal-sounding phrasing with more natural training-objective prose. |
| `20260621_1013_bdl_language_status_cleanup.pdf` | Tightened abstract/method terminology, reduced repeated event-mass prose, and recorded the latest P9 Transformer status. |
| `20260621_1916_bdl_transformer_diagnostic_appendix.pdf` | Added appendix-ready offline-Transformer diagnostic table and interpretation after P9 completed. |
| `20260621_1924_bdl_language_stride_cleanup.pdf` | Broadened and shortened the abstract, sharpened localized-supervision novelty, aligned architecture-control wording, and justified one-second CHB-MIT stride. |
| `20260621_1928_bdl_figure_terminology_cleanup.pdf` | Regenerated figures after changing the method panel and captions from generic event-curve wording to binned-count and score-sequence terminology. |
| `20260621_1930_bdl_vocabulary_consistency_cleanup.pdf` | Removed remaining event-curve/event-target drift in protocol, results, discussion, and appendix text. |
| `20260621_1946_bdl_prose_consistency_cleanup.pdf` | Smoothed abstract, introduction, results, and appendix language around event-count supervision, ablation scope, and the main objective comparison. |
| `20260621_1950_bdl_appendix_protocol_cleanup.pdf` | Removed paper-facing references to unfinished fine-stride settings and made appendix protocol/ablation labels read as studies and controls rather than internal diagnostics. |
| `20260621_2004_bdl_broader_prose_ablation_cleanup.pdf` | Broadened the abstract and introduction, made ablation narration less checklist-like, simplified discussion terminology, and tightened manuscript vocabulary checks. |
| `20260621_2040_bdl_conclusion_relatedwork_cleanup.pdf` | Smoothed conclusion, related-work, and discussion phrasing around counted events, localized regression, and count-preserving targets. |
| `20260621_2058_architecture_figure_cleanup.pdf` | Clarified architecture controls versus Transformer diagnostics, refreshed P11 status, and fixed compact figure labels to avoid plot-text collisions. |
| `20260621_2105_abstract_relatedwork_prose.pdf` | Made the abstract opener broader and more natural, replaced awkward counted-event wording with event-occurrence phrasing, and refreshed P11 status. |
| `20260621_2209_vocab_prose_ops_cleanup.pdf` | Removed remaining compressed mass/count-preservation phrasing from paper prose, aligned vocabulary guardrails, and refreshed P11 Trace status. |
| `20260621_2240_ablation_vocab_cleanup.pdf` | Replaced internal-sounding ablation and diagnostic language with reader-facing control/scope wording, aligned the event-count derivation title, and refreshed P11 status. |
| `20260621_2244_abstract_conclusion_story_cleanup.pdf` | Reworked the abstract, discussion opening, and conclusion around target/metric alignment and event occurrences while keeping empirical claims fixed. |
| `20260621_2246_related_work_novelty_cleanup.pdf` | Sharpened the related-work distinction between localized supervision and BDL's event-occurrence unit plus Poisson log score; cleaned matching vocabulary notes in this ops file. |
| `20260621_2253_sleep_main_figure_balance.pdf` | Rebalanced the main sleep-results figure layout and compacted the AP legend while preserving the data and minimal panel text. |
| `20260621_2258_sleep_result_caption_cleanup.pdf` | Expanded the sleep-results caption to state the objective and backbone controls directly, and refreshed P11 after fold 2 completed. |
| `20260621_2306_unit_sum_ablation_prose_cleanup.pdf` | Standardized smoothing language to unit-sum kernels, clarified architecture-control and state-reconstruction ablations, expanded hazard recursion explanation, and refreshed P11 status. |
| `20260621_2309_relatedwork_novelty_cleanup.pdf` | Reframed localized-supervision related work around BDL's output-timeline unit and scoring rule, smoothed intro/results occurrence wording, and refreshed P11 status. |
| `20260621_2312_discussion_scope_cleanup.pdf` | Made the discussion opening more direct about BDL as a training objective, clarified smoothing under clipping/downsampling, and refreshed P11 epoch-10 status. |
| `20260621_2317_claim_language_guardrail.pdf` | Replaced remaining reviewer-instruction phrasing in method and appendix text with direct claims, and added checker guardrails for those phrases. |
| `20260621_2327_abstract_intro_cleanup.pdf` | Smoothed the abstract, introduction, related work, results, and conclusion; extended the manuscript checker to catch both positive and negative read-as phrasing; refreshed P11 after the Gaussian row completed training. |
| `20260621_2332_ablation_argument_cleanup.pdf` | Reworked appendix transfer, target, context, and point-event commentary so ablations read as tests of alternative explanations rather than a checklist of extra runs; refreshed P11 scorer and segmentation-train status. |
| `20260621_2335_appendix_language_polish.pdf` | Replaced remaining operations-sounding appendix phrasing around transfer controls, Transformer controls, and state reconstruction with paper-facing analysis language; refreshed P11 scorer and segmentation-train status. |
| `20260621_2337_results_control_language.pdf` | Smoothed results and experiment prose around architecture controls and appendix evidence, reducing row-ledger phrasing while keeping claims and numbers fixed; refreshed P11 scorer and segmentation-train status. |
| `20260621_2344_figure_label_markdown_policy.pdf` | Replaced abbreviated attention-gated U-Net figure labels, regenerated the main sleep-results figure, enforced the two-file source Markdown policy, and refreshed P11 scorer and segmentation-train status. |
| `20260621_2348_protocol_table_stride_clarity.pdf` | Added a compact protocol table, clarified CHB-MIT post-training rescoring on the 256 Hz timeline, and tightened stride-language around one-second and two-second CHB-MIT settings. |
| `20260621_2352_secondary_wording_trace_observability.pdf` | Replaced remaining primary/secondary-style appendix phrasing, added a checker guardrail for that wording, and recorded the P11 Trace live-progress observability issue. |
| `20260622_0001_results_conclusion_trace_status.pdf` | Smoothed results and conclusion language around the objective comparison and appendix controls, refreshed P11 status, and recorded the current strict-scorer Trace observability gap. |
| `20260622_0007_state_reconstruction_appendix.pdf` | Expanded the state-reconstruction appendix to explain peak filling, cumulative evidence, and hazard calibration without promoting reconstruction to a main claim; refreshed P11 status. |
| `20260622_0010_ablation_story_cleanup.pdf` | Reduced repeated target/setting vocabulary in results and experiments, made the ablation narrative more reviewer-facing, and refreshed P11 after fold-0 epoch-20 validation. |
| `20260622_0014_discussion_scope_cleanup.pdf` | Tightened the discussion scope statement around AP-aligned training, segmentation, and detector design choices; refreshed P11 fold-1 progress. |
| `20260622_0021_broader_prose_scope_cleanup.pdf` | Broadened the abstract opening, made the results scope language less ledger-like, clarified the discussion's detector-scope statement, and refreshed P11 fold-1 progress. |
| `20260622_0026_protocol_point_event_cleanup.pdf` | Expanded protocol commentary for matched optimizer/checkpoint control, clarified point-event segmentation baselines, and refreshed P11 fold-1 progress. |
| `20260622_0031_paired_fold_evidence.pdf` | Added a generated paired-fold consistency table and a concise Results reference showing the main gains are consistent across matched validation records; refreshed P11 fold-1 progress. |
| `20260622_0042_event_occurrence_reference_cleanup.pdf` | Smoothed event-occurrence wording, fixed the paired-fold table reference, refreshed P11 fold-1 status, and recorded the remaining-row Trace queue observability note. |
| `20260622_0047_vocabulary_guardrail_cleanup.pdf` | Reduced repeated target terminology in headline and appendix prose, added manuscript guardrails for reverted vocabulary, and refreshed P11 fold-2 status. |
| `20260622_0050_paired_sign_check.pdf` | Added exact one-sided sign-check probabilities to the paired-fold consistency table and refreshed P11 fold-2 epoch-5 status. |
| `20260622_0055_claim_guardrail_cleanup.pdf` | Softened overbroad abstract/results phrasing, added overclaim guardrails, and refreshed P11 fold-2 epoch-8 status. |
| `20260622_0103_abstract_count_language_cleanup.pdf` | Broadened the abstract opener, replaced compressed event-count wording in headline prose, rendered a fresh review PDF, and refreshed P11 fold-2 epoch-14 status. |
| `20260622_0106_abstract_ablation_claim_cleanup.pdf` | Replaced the abstract's ablation checklist with a clearer claim about ruled-out explanations and detector-design limits; refreshed P11 fold-2 epoch-16 status. |
| `20260622_0109_internal_language_guardrail.pdf` | Replaced operational manuscript phrasing such as learning-rate screens and experiment records with paper-facing pilot/configuration language, added checker guardrails for those phrases, and refreshed P11 fold-2 epoch-18 status. |
| `20260622_0121_conclusion_trace_status_refresh.pdf` | Broadened the conclusion opening around the training/evaluation mismatch, replaced an awkward discussion phrase about binned objectives, recorded the current P11 Trace observability state, and refreshed the review PDF. |
| `20260622_0124_vocabulary_appendix_cleanup.pdf` | Replaced clunky binned event-occurrence phrasing in captions and appendix prose with expected-events-per-output-bin wording, added a manuscript guardrail for that vocabulary drift, and refreshed P11 fold-3 status. |
| `20260622_0127_experiment_prose_cleanup.pdf` | Reworked experiment and appendix wording from run-ledger phrasing toward configuration/comparison language, refreshed P11 fold-3 epoch-9 status, and rendered a new review PDF. |
| `20260622_0130_appendix_row_language_cleanup.pdf` | Removed remaining row-ledger phrasing from appendix prose and captions, aligned P9 Transformer wording with architecture-control framing, and refreshed P11 fold-3 status. |
| `20260622_0136_figure_label_cleanup.pdf` | Regenerated the method overview with the expected-events panel label, confirmed the two-file source Markdown policy, and refreshed P11 fold-3 epoch-16 status. |
| `20260622_0141_protocol_prose_cleanup.pdf` | Replaced remaining matrix/pilot/record-style manuscript phrasing with paper-facing protocol language, added guardrails against those regressions, and refreshed P11 fold-3 epoch-19 status. |
| `20260622_0316_abstract_status_cleanup.pdf` | Split the abstract's dense ablation sentence into clearer ruled-out-explanation and detector-scope statements, refreshed P11 Trace status, and confirmed the two-file source Markdown policy. |
| `20260622_0321_stride_result_cleanup.pdf` | Connected the CHB-MIT stride result to the metric-resolution heuristic, clarified that kernel choice changes with output stride, and recorded the partial strict-scorer Trace artifact issue. |
| `20260622_0327_conclusion_status_cleanup.pdf` | Removed repeated method jargon from the conclusion, refreshed P11 U-Net progress, and kept the current review PDF synchronized with source. |
| `20260622_0340_language_status_cleanup.pdf` | Broadened the abstract opening around time-series decisions, reduced repeated state-occupancy phrasing in the main text, refreshed P11 U-Net fold status, and rendered a clean review PDF. |
| `20260622_0343_main_figure_cleanup.pdf` | Regenerated the main sleep-results figure with the AP legend outside the data region, quieter markers/grid, refreshed P11 fold-1 status, and rendered a clean review PDF. |
| `20260622_0410_transformer_hook_architecture_language.pdf` | Added a conditional generated table hook for the tuned Transformer candidate, clarified why self-attention evidence is separated from the convolutional architecture controls, and refreshed P11/P15 status. |
| `20260622_0413_related_work_transformer_status.pdf` | Sharpened the related-work contrast with universal regression/proximity-target framing, refreshed P11/P15 status, and rendered a clean review PDF. |
| `20260622_0428_transformer_candidate_queue.pdf` | Added the stronger offline-Transformer candidate plumbing, queued the P16 Trace training/scoring waiters, kept the Transformer table conditional, and rendered a clean review PDF. |
| `20260622_0433_transformer_review_policy.pdf` | Kept the Transformer comparison as a reviewed conditional table rather than an automatic main-row replacement, cleaned paper-facing self-attention wording, and refreshed P11/P15/P16 status. |
| `20260622_0438_positive_scope_language.pdf` | Replaced defensive negative phrasing in the abstract, method, results, discussion, and appendix with direct scope statements; refreshed P11/P15/P16 status. |
| `20260622_0456_transformer_main_row_policy.pdf` | Made the strong Transformer the conditional replacement for the attention-gated U-Net main architecture row when the matched BDL-Hard comparison is favorable, retained attention-gated U-Net as an appendix sensitivity check, and recorded the P15 scorer failure/P16 early-start Trace lessons. |

## Acceptance Checks

Run the relevant subset before handing off a paper or experiment change:

```bash
git diff --check
uv run --offline python -m py_compile paper/make_plots.py paper/collect_results.py paper/experiment_status.py paper/run_experiments.py paper/score_segmentation_reconstruction.py paper/paired_fold_analysis.py paper/check_data.py src/utils.py train.py train_all.py eval.py models/load_model.py models/OnlineModels.py
uv run --offline python paper/check_manuscript.py
uv run --offline python -m unittest tests.test_objectives tests.test_models tests.test_postprocessing tests.test_experiment_status
uv run --offline python paper/check_data.py --datasets sleep seizure bowshock fraud --sleep-dir data/sleep --seizure-dir data/seizure --bowshock-dir data --fraud-dir data
.tools/tectonic/tectonic -X compile paper/main.tex --outdir paper/build --keep-logs --keep-intermediates
```

Also check that generated logs, checkpoints, cached predictions, LaTeX build
files, and experiment outputs are not staged.
