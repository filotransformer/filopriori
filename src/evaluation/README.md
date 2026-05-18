# Evaluation Module

This module contains all evaluation and metrics calculation functionality for the Filo-Priori V7 project.

## Core Modules

### apfd.py - APFD Calculation (Consolidated)

**Single source of truth for all APFD calculations.**

This module consolidates all APFD functionality (previously split between apfd.py and apfd_calculator.py).

**Key Functions:**

1. **count_total_commits(df_build)** - Count unique commits including CRs
2. **calculate_apfd_single_build(ranks, labels)** - Calculate APFD for one build
3. **calculate_apfd_per_build(df, ...)** - Calculate APFD for all builds
4. **calculate_ranks_per_build(df, ...)** - Calculate priority ranks per build
5. **generate_apfd_report(df, ...)** - Generate complete APFD report
6. **generate_prioritized_csv(df, ...)** - Generate prioritized test cases CSV
7. **print_apfd_summary(summary_stats)** - Print formatted summary

**Critical Business Rules:**

1. **count_tc=1 → APFD=1.0** - When only 1 test case, APFD is always 1.0
   - Rationale: No ordering optimization is possible with 1 TC
   - Reference: master_vini/src/evaluation/apfd_calculator.py
   - Documentation: /docs/TC_COUNT_FIX.md

2. **Only Fail results** - Only builds with at least 1 "Fail" result are included
   - Other statuses (Blocked, Delete, Pass) are excluded
   - Expected result: Exactly 277 builds

3. **Per-build calculation** - APFD is calculated PER BUILD, not globally
   - Each build gets its own APFD score
   - Final metric is the mean APFD across all builds

**Usage Example:**

```python
from src.evaluation.apfd import (
    calculate_ranks_per_build,
    generate_apfd_report,
    print_apfd_summary
)

# Calculate ranks
df = calculate_ranks_per_build(df, probability_col='probability', build_col='Build_ID')

# Generate APFD report
results_df, summary_stats = generate_apfd_report(
    df, 
    method_name="dual_stream_gnn_exp_17",
    test_scenario="full_test_csv_277_builds",
    output_path="results/experiment_017/apfd_per_build.csv"
)

# Print summary
print_apfd_summary(summary_stats)
```

**Output Format:**

CSV with columns:
- method_name: Prioritization method name
- build_id: Build identifier
- test_scenario: Test scenario type
- count_tc: Number of unique test cases in build
- count_commits: Number of unique commits (including CRs)
- apfd: APFD score (0.0 to 1.0, higher is better)
- time: Processing time (placeholder: 0.0)

### metrics.py - Classification Metrics

Standard classification metrics (accuracy, precision, recall, F1, etc.)

## Archived Modules

Previous versions and redundant modules are archived in `/archive/evaluation/`:
- `apfd_calculator.py` - Archived APFDCalculator class (consolidated into apfd.py)

See `/archive/evaluation/README.md` for details.

## Related Scripts

- `/scripts/recalculate_apfd_fix_count_tc_1.py` - Recalculate APFD with count_tc=1 fix

## Documentation

- `/docs/TC_COUNT_FIX.md` - Documentation of count_tc=1 → APFD=1.0 fix
- `/archive/evaluation/README.md` - Archive documentation

---
**Last Updated:** 2025-11-06
**Maintainer:** Filo-Priori V7 Team
