# Paper Next Steps

This file separates work needed for a credible workshop submission from follow-up analyses that could become a stronger later version. The current figures and tables are placeholders until the experiment matrix is rerun from this branch.

## Submission-Critical

- Rerun the sleep experiments from scratch with `paper/run_experiments.py`.
- Compare `density_hard`, `density_gau`, and `density_custom` against legacy MSE objectives `hard`, `gau`, `custom` and segmentation objectives `seg`, `seg_weighted`, and `seg_focal`.
- Run the online-context ablation: bidirectional `gru` versus `fgru`, `flstm`, and `causal_transformer`.
- Keep the alternating onset/offset decoder in the interval post-processing sweep and report whether it helps.
- Report fold means and standard deviations for event AP; use pooled EDAP curves only as secondary diagnostics.
- Regenerate final plots from the Kaggle `sleep-plots.ipynb` workflow or `paper/make_plots.py` after reruns finish.
- Compile the paper with `latexmk` and commit only source files, not build products, checkpoints, cached predictions, generated scripts, or disposable figures.

## Writing Next Steps

- Keep the paper framed around maximum-likelihood boundary density estimation, not MSE heatmap regression.
- State that the Poisson objective estimates independent onset and offset boundary rates.
- Treat legal sleep/wake alternation as a decoder or post-processing constraint, not something guaranteed by the likelihood.
- Keep online models as an ablation unless the results are strong enough to support a main claim.
- Expand prior-work coverage where needed: heatmap regression, temporal boundary localization, event AP metrics, segmentation baselines, and temporal point processes.
- Avoid abstract claims about online models or alternating decoding until the rerun results justify them.

## Future Analysis: Boundary Predictions as Segmentation

It would be useful to convert onset/offset boundary predictions into binary state predictions and score them with traditional segmentation metrics. This would answer a different question from event AP: not just whether boundaries are localized within tolerance, but whether the induced sleep/awake state sequence is useful.

Proposed decoder:

1. Generate onset and offset candidates from a density model.
2. Apply the alternating onset/offset decoder.
3. Choose an operating threshold on validation folds, separately from the event-AP threshold if needed.
4. Convert each retained onset-offset pair into a binary interval: sleep is 1 between onset and offset, awake is 0 elsewhere.
5. Handle incomplete intervals deterministically: drop leading offsets, drop trailing onsets, and document whether day-boundary carryover is allowed.

Metrics to add:

- per-timestep accuracy and balanced accuracy;
- macro F1 for sleep versus awake;
- sleep-class precision, recall, and F1;
- Dice and Jaccard/IoU for predicted sleep intervals;
- duration error per night or per recording;
- transition-count error as a sanity check for over-fragmentation.

Comparisons:

- density model plus alternating decoder versus `seg`, `seg_weighted`, and `seg_focal`;
- unconstrained boundary peaks converted to intervals versus alternating-decoder intervals;
- bidirectional versus online models under the same interval decoder.

Framing caveat:

This should be presented as a downstream reconstruction analysis, not as the main objective. Boundary Density Likelihood is optimized for ranked boundary events. If segmentation metrics dominate the story, the honest conclusion may be that density likelihood is better suited to event detection while segmentation losses remain better suited to state occupancy.

## Later Extensions

- Add a learned interval decoder or lightweight dynamic program that jointly scores onset/offset candidates and interval durations.
- Add duration priors for sleep intervals and seizure intervals, evaluated separately from the core likelihood objective.
- Try calibration plots for predicted boundary rates: expected count versus observed count per recording.
- Add confidence intervals over seeds once the full matrix is stable.
- Consider seizure-specific interval reconstruction only after the sleep workflow is validated.
