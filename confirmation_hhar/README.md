# Additional HHAR confirmation

This study is complete. Its original protocol was publicly committed before preprocessing outcomes, model fitting or predictive results (commit4cf2155, 14 September2026). It uses nine users, 10,800 windows, five seeds, three disjoint-user folds and the unchanged LR/MR/IT models: 45 classifier cases. Main-study model cases remain separate.

## Primary result

LAC, N=240, delta=0.05, tau=0.5; 40 paired query orders per model case:

| Method | Saved reference labels | Threshold failure | Inflation failure |
|---|---:|---:|---:|
| JRC | 34.03% | 1.33% | 0.28% |
| Bonferroni HG | 28.58% | 0.39% | 0.00% |
| CS Beta(9,1) | 25.13% | 0.28% | 0.00% |
| Uniform-prior CS | 20.64% | 0.06% | 0.00% |

The fixed stronger prior was specified before HHAR outcomes. Paired participant-bootstrap differences favor JRC by5.45percentage points versus HG (95% CI5.09–5.85) and8.89versus Beta(9,1) CS (8.52–9.23), conditional on these models/partitions. See summaries/ for all primary and secondary outcomes, per-user results and classifier metrics. These intervals are not general population guarantees.

## Reproduce

From the repository root install requirements.txt and run `python get_artifacts.py`. Run `python confirmation_hhar/reproduce.py --smoke` for one case or omit `--smoke` for all45. The full45-case replay already matched the frozen results (full_replay_check.json). `python boundary_demo.py --pool-size 240` independently reconstructs the label-free boundary table.

For preprocessing obtain Phones_accelerometer.csv from the [original UCI dataset](https://archive.ics.uci.edu/dataset/344/hhar), then run `python confirmation_hhar/prepare.py --csv PATH`. Use a separate working copy for new preprocessing/refits; cached outputs are not silently replaced. Additional classifier dependencies are in revision6/requirements.txt. The declared IT run requires the original CUDA-enabled environment. Training: `python confirmation_hhar/train.py`; acquisition: `python confirmation_hhar/evaluate.py`; aggregation: `python confirmation_hhar/analyze.py`.

## Scope and data quality

The public cohort was not used in the earlier seven-dataset model study. This is a public project protocol, not an independent-registry registration. No human annotation-time experiment is claimed. The fixed timestamp-continuity rule retains only five Nexus4 windows before the per-user cap and few s3mini_2 windows. Device-specific counts are in data/metadata.json and data_audit.json. No exact duplicate retained windows or cross-user duplicate groups were found. These exclusions are reported without changing the protocol after results.

Original data license: CC BY4.0; dataset DOI10.24432/C5689X. Raw CSV files are not redistributed; their hashes and derived windows/probabilities are provided for reproducibility.
