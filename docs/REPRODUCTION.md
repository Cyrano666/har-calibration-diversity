# Reproduce the study

Use Python 3.12 and install the root requirements.txt. Download the checksum-verified Online Resource 1 with `python get_artifacts.py`. A local copy can be installed with `python get_artifacts.py --archive PATH_TO_ZIP`. Existing files are retained when their bytes match; conflicting files stop extraction for review.

## Acquisition from saved probabilities

- `python reproduce.py --smoke`: one complete case, compared with frozen rows.
- `python reproduce.py`: all 354 cases and the aggregate tables.
- `python revision6/reproduce.py --resume --output PATH`: resume a timestamped output directory.
- `python revision7/prior_audit.py --full`: reconstruct the subsequent four-prior diagnostic.
- `python revision7/figures.py`: reconstruct quantitative figures; the editable architecture is supplied under figures/.

Reproductions create timestamped output directories and preserve the locked reference results. These commands do not fit new classifiers and do not establish an independent confirmation experiment.

## Refitting

Additional classifier dependencies are in revision6/requirements.txt. Training and preprocessing scripts are provided under revision/, revision5/ and revision6/. Obtain the original signals from their providers as described in docs/DATA.md. Raw USC-HAD signals are not redistributed. The supplied original classifier implementation used an RTX 4070 Laptop GPU and the pinned CUDA-enabled training environment; CPU timing is not comparable. Use a separate working copy for refits, because training scripts cache completed cases.

## Evidence map

| Manuscript item | Frozen source |
|---|---|
| Table 1 | revision7/literature_comparison.md |
| Table 2 | Dataset metadata and cited providers |
| Table 3, Figures 2 and 5 | revision6/summaries/overview.csv |
| Table 4, Figure 3 | revision6/summaries/failure_rates.csv and overview.csv |
| Table 5 | revision6/results/predictions/*.json and model_summary.csv |
| Table 6 | Main overview and failure summaries |
| Table 7 | revision7/prior_full_summary.csv |
| Table 8 | revision6/runtime.csv |
| Figure 4 | revision6/summaries/paired_intervals.csv |
| Figure 6 | revision6/results/predictions/uschad_0_0_MR.npz |
| Figure 7 | revision6/summaries/model_summary.csv |
| Ambiguous trial exclusion | revision6/schema_sensitivity/ |

Compile paper/latex/manuscript.tex with pdfLaTeX three times. The unmodified official Springer Nature class and bibliography style are included. The manuscript is an author-review draft, not an accepted publication.
