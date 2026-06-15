# Density-Likelihood Rewrite TODO

## Submission blockers

- Replace legacy MSE-centered claims with a maximum-likelihood boundary-density framing.
- Rerun experiments from scratch; current tables and plots are placeholders only.
- Compile the rewritten paper with a clean LaTeX preamble and no generated artifacts committed.
- Add prior-work coverage for heatmap regression, temporal boundary localization, event metrics, and temporal point processes.

## Core experiments

- Sleep dataset first: compare `density_hard`, `density_gau`, and `density_custom` against legacy `hard`, `gau`, `custom`, and segmentation `seg1`/`seg2`.
- Seizure dataset second if compute allows, using the same objective matrix.
- Report fold means and standard deviations, plus pooled EDAP only as a secondary view.
- Add likelihood-vs-MSE, kernel-type, and prior-rate ablations.

## Plot regeneration

- Treat all current paper figures as disposable.
- Regenerate final plots from the Kaggle `sleep-plots.ipynb` workflow after the new runs finish.
- Prefer plots that explain the new likelihood framing: boundary targets, learned rates, peak extraction, and EDAP-by-tolerance curves.

## Writing and framing

- Rename the method around boundary density/intensity, not generic PDF regression.
- State that temporal point processes are related but heavier than needed for this supervised boundary-localization setting.
- Keep claims dataset-specific and metric-specific.
- Make limitations explicit: segmentation can still win when state occupancy is the desired output.
