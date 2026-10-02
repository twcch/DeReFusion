# 24c · C1 blind handoff dependency addendum

**Date:** 2026-09-18  
**Status:** integrity repair only; no statistic, threshold, cohort, seed, feature
definition or verdict rule is changed.

The independent Stage-1 reviewer verified all 265 sealed inputs in commit
`780b77b89377ac5e2a70905b10b53e9ca06115fa`, then stopped because the frozen
`c1_predictor.py` imports `structure_routing_experiment.window_features`, while
that exact dependency was missing from the result-free manifest. The reviewer
did not choose an alternative ACF1 definition and did not form a verdict.

The repair is limited to adding the exact
`reproduction/analysis/structure_routing_experiment.py` used by the analyst.
The only contemporaneous edit in that module relocates output files from a
machine-specific sibling directory to `reproduction/results/`; it does not
change `window_features`, any split, feature, seed, statistic or threshold.

Stage 1 must restart at the complete hash gate. The new manifest contains 267
files: the previous 265 plus this addendum and the missing dependency. The
earlier failed handoffs remain part of the audit record.
