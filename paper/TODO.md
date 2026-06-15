# Density-Likelihood Rewrite TODO

## Submission blockers

- Replace legacy MSE-centered claims with a maximum-likelihood boundary-density framing.
- Rerun experiments from scratch; current tables and plots are placeholders only.
- Compile the rewritten paper with a clean LaTeX preamble and no generated artifacts committed.
- Add prior-work coverage for heatmap regression, temporal boundary localization, event metrics, and temporal point processes.

## Core experiments

- See `paper/NEXT_STEPS.md` for the current submission path and future analyses that should not block the first workshop version.
- Sleep dataset first: compare `density_hard`, `density_gau`, and `density_custom` against legacy `hard`, `gau`, `custom`, and segmentation `seg`, `seg_weighted`, and `seg_focal`.
- Seizure dataset second if compute allows, using the same objective matrix.
- Report fold means and standard deviations, plus pooled EDAP only as a secondary view.
- Add likelihood-vs-MSE, kernel-type, and prior-rate ablations.
- Add an online-model ablation: compare bidirectional `gru` against `fgru`, `flstm`, and FlashAttention-backed `causal_transformer`.
- Include alternating onset/offset post-processing in the post-processing ablation, matching the competition-style sleep/wake cleanup.
- Use a fixed seed list for reruns; start with `--seed 0` to match the legacy split, then add two more seeds if runtime is acceptable.
- Keep raw experiment outputs under `experiments/`; do not commit model checkpoints, cached predictions, or generated plots.
- Prefer `paper/run_experiments.py` for reruns so command lines and matrix membership stay consistent.
- Use `paper/EXPERIMENTS.md` as the GPU-later runbook; generate shell scripts with `--write-script` when moving to Kaggle or a GPU box.
- Use the structured files in `experiments/*/*/*/seed_*/results/` for tables and plots.

## Suggested rerun commands

Run the full objective matrix on sleep first:

```bash
python train_all.py --dataset sleep --datadir data --epochs 20 --folds 4 --normalize True
```

If runtime is too high, run a focused model/objective matrix first:

```bash
python train.py --dataset sleep --model gru --objective density_custom --datadir data --epochs 20 --folds 4 --normalize True
python train.py --dataset sleep --model gru --objective custom --datadir data --epochs 20 --folds 4 --normalize True
python train.py --dataset sleep --model gru --objective seg --datadir data --epochs 20 --folds 4 --normalize True
python train.py --dataset sleep --model gru --objective seg_weighted --datadir data --epochs 20 --folds 4 --normalize True
python train.py --dataset sleep --model gru --objective seg_focal --datadir data --epochs 20 --folds 4 --normalize True
```

Then repeat the strongest density objective and strongest baseline on seizure:

```bash
python train.py --dataset seizure --model unet --objective density_custom --datadir data --epochs 20 --folds 4 --normalize True
python train.py --dataset seizure --model unet --objective seg --datadir data --epochs 20 --folds 4 --normalize True
```

These are starting commands, not final hyperparameter claims. Record the exact command, commit hash, data source, and runtime for every table.

## Plot regeneration

- Treat all current paper figures as disposable.
- Regenerate final plots from the Kaggle `sleep-plots.ipynb` workflow after the new runs finish.
- Prefer plots that explain the new likelihood framing: boundary targets, learned rates, peak extraction, and EDAP-by-tolerance curves.
- Before committing regenerated plots, strip notebook outputs or export static figures into `paper/figures/` with descriptive filenames.
- Prefer `python paper/make_plots.py --dataset sleep --results-root experiments` for final paper figures.
- Replace any legacy plot that implies MSE/PDF regression is the main method.
- Required plot set:
  - target kernels for `density_hard`, `density_gau`, and `density_custom`;
  - predicted onset/offset rates with detected peaks;
  - EDAP by tolerance for density, MSE, and segmentation objectives;
  - ablation plot for likelihood vs MSE under the same kernel.

## Writing and framing

- Rename the method around boundary density/intensity, not generic PDF regression.
- State that temporal point processes are related but heavier than needed for this supervised boundary-localization setting.
- Keep claims dataset-specific and metric-specific.
- Make limitations explicit: segmentation can still win when state occupancy is the desired output.
- Do not claim density likelihood wins until reruns support it.
- Use "legacy MSE heatmap" for the old objective family.
- Use "tolerance-aligned kernel" for `density_custom`; avoid saying it directly optimizes AP.
- Describe online ablations as streaming-context stress tests, not as the main architecture contribution.
- State clearly that the likelihood emits independent onset/offset rates; legal interval alternation is a decoder/post-processing constraint.

## Related-work additions to triage

Searched June 15, 2026. Add only the references that support a sentence we actually need in the paper.

- Heatmap regression foundations: add one or two canonical pose-estimation heatmap papers, such as Newell et al. "Stacked Hourglass Networks for Human Pose Estimation" and Xiao et al. "Simple Baselines for Human Pose Estimation and Tracking." Use these to establish that Gaussian/soft heatmaps are a standard localization target, then keep Luo et al. and Yu et al. for quantization/heatmap-analysis details.
- Heatmap encoding bias: consider Huang et al. "The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation" if we discuss annotation-to-grid encoding and decoding bias. This is relevant to our finite-window kernel normalization and peak-decoding details.
- Temporal boundary localization: add Lin et al. "BSN: Boundary Sensitive Network for Temporal Action Proposal Generation" and/or Lin et al. "BMN: Boundary-Matching Network for Temporal Action Proposal Generation." Use them as adjacent work on learned start/end boundary probabilities and boundary pairing, while making clear that video action proposals are not our physiological time-series setting.
- Event and range metrics: keep SoftED as the closest event-detection metric reference, and add Tatbul et al. "Precision and Recall for Time Series" if we discuss range-based/event-aware precision-recall beyond point AP. Consider Lavin and Ahmad's NAB paper only if we discuss streaming anomaly scoring or early-detection rewards.
- Sleep from accelerometers: add van Hees et al. work on accelerometer sleep-period detection, especially HDCZA / sleep-period time-window estimation, to ground the Child Mind sleep task in the sleep-wearables literature. Keep GGIR as tooling/background rather than a method baseline.
- Temporal point processes: current RMTPP and neural TPP survey citations are useful; add Mei and Eisner "The Neural Hawkes Process" if we need a canonical neural intensity-process reference, and Omi et al. "Fully Neural Network based Model for General Temporal Point Processes" if we discuss intensity parameterization and likelihood integration. In all cases, state that our model is a supervised conditional boundary-rate objective, not a full event-history generative model.
- Online/streaming localization: for the online ablation, cite a small amount of adjacent online temporal action localization work, such as Kim et al. "A Sliding Window Scheme for Online Temporal Action Localization," Xu et al. "Long Short-Term Transformer for Online Action Detection," or Song et al. "Online Temporal Action Localization with Memory-Augmented Transformer." Use this only to justify the streaming constraint, not to imply we solve video action localization.
- Competition-specific sleep cleanup: keep the Kaggle competition citation for EDAP and alternating onset/wakeup submission constraints; avoid relying on forum solution writeups unless we explicitly cite them as competition practice rather than peer-reviewed prior work.

## Acceptance checklist

- `python3 -m py_compile src/utils.py train.py train_all.py eval.py models/load_model.py` passes.
- `python paper/run_experiments.py --datasets sleep --models gru --objectives density_custom --seeds 0 --epochs 1 --folds 2 --datadir data` prints a valid dry-run command.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` passes from `paper/`.
- No LaTeX build artifacts, model checkpoints, cached predictions, or experiment outputs are staged.
- The paper has no tables or plots from old runs unless they are explicitly labeled placeholders.
