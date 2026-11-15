# Benchmark Result Tables

This directory contains the evaluation summary tables reported in the paper across all nine task groups (`T1`–`T9`), 27 leave-one-fault-out configurations (`Tn-U1`..`U3`), and three random seeds (`{0, 1, 2}`):

- `e1_group_summary.csv`: Complete-method comparison across `T1`–`T9` (Tables II–IV, unweighted mean and sample standard deviation over $n = 9$ runs per group; metrics in `[0, 1]`).
- `e1_task_summary.csv`: Instance-level evaluation across all 27 tasks (`T1-U1` through `T9-U3`, $n = 3$ seeds per instance; Supplementary Table S6).
- `e2_scoring_summary.csv`: Fixed-encoder alternative scoring rules (`MSP`, `Energy`, `Static prototype`, `Online Orbit` on the OrbitFD encoder; Table V) and common-scorer comparisons across methods.
- `e3_ablation_runs.csv`: Run-level mechanistic ablation metrics (`Static prototype`, `Normal-only`, `Frozen-local Orbit`, `Frozen-global Orbit`, `Source-context union`, `Ungated causal`, `Online Orbit`; Table VI).
- `e4_cpu_efficiency.csv`: Single-thread desktop CPU latency (`batch=1` and `batch=32`), parameter counts, memory footprint, and dense MACs/FLOPs (Section V-D).

## Metric Conventions

- **`known_acc` ($\uparrow$)**: Known-class classification accuracy before rejection.
- **`auroc` ($\uparrow$)**: Area under the ROC curve with unknown faults as the positive class.
- **`aupr` ($\uparrow$)**: Average precision with unknown faults as the positive class.
- **`fpr95` ($\downarrow$)**: Fraction of unknown windows accepted at the first empirical threshold retaining at least 95% of known windows (all score ties retained).
