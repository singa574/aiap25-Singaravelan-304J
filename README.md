# AIAP Batch 25 - Travelers' Inn booking-fallout prediction

**Candidate name (as in NRIC):** TODO  
**Application email:** TODO

> This is a submission-ready README outline. Replace every `TODO` with your own results and wording after running the EDA and pipeline.

## Problem statement

At the time a reservation is confirmed, estimate the probability that it will **fail to materialise as an occupied stay because it is cancelled or becomes a no-show**, rather than ending in check-out. Revenue and Reservations can use the probabilities to rank upcoming bookings for low-harm interventions such as reminders, reconfirmation, or policy review, and to inform carefully controlled overbooking decisions without adding rooms or staff.

The target is binary: `is_canceled = 1` (which the data confirms covers both `Canceled` and `No-Show`) versus `is_canceled = 0` (`Check-Out`). The pipeline produces probabilities and a ranked 2025 holdout list; it does not automate cancellation, pricing, or denial of service.

### Why this framing

- Empty-room risk is the stakeholder's clearest stated pain and is directly labelled in the booking table.
- No-shows alone are only about 1% of raw records, so a no-show-only model would be much harder to validate reliably. Cancellation plus no-show is about 36% and represents the shared business outcome: a reserved room does not become an occupied stay.
- Price optimisation needs demand, inventory, competitor-price, and willingness-to-pay information that is not present. Reviews are post-stay data and cannot predict whether that same stay will occur.
- The prediction cutoff is reservation confirmation. This makes the output actionable and defines which fields are legal predictors.

## Repository structure

```text
.
├── .github/workflows/github-actions.yml  # supplied assessment workflow
├── config/default.yaml                   # data, split, preprocessing, model, evaluation
├── data/.gitkeep                         # put hotel.db here locally; DB is ignored
├── src/
│   ├── config.py                         # config loading and CLI overrides
│   ├── data.py                           # read-only SQLite ingestion and cleaning
│   ├── features.py                       # booking-time feature construction
│   ├── modeling.py                       # preprocessing + estimator factory
│   ├── evaluate.py                       # metrics and validation threshold
│   └── train.py                          # end-to-end orchestration and artefacts
├── tests/test_data.py                    # cleaning and leakage guard tests
├── eda.ipynb                             # presentation-style EDA plan/notebook
├── run.sh                                # assessment entry point
├── requirements.txt                      # pip dependencies
├── decision_log.md                       # supplied five-question template
├── prompt_chat_history.md                # supplied transcript template
└── SUBMISSION_CHECKLIST.md               # final QA checklist
```

Generated files go to `artifacts/` and are ignored by Git.

## Setup and execution

1. Use Python 3.10 or newer and create a virtual environment.
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Place the supplied database at `data/hotel.db`. Do not add it to Git.
4. Run the pipeline from the repository root: `bash run.sh`.
5. Run checks: `python -m unittest discover -s tests`.

Examples of configuration overrides:

```bash
bash run.sh --model random_forest --output-dir artifacts/random_forest
bash run.sh --db-path /absolute/local/path/hotel.db
```

For repeatable experiments, copy `config/default.yaml`, edit model parameters or split years, and run `python -m src.train --config config/my_experiment.yaml`. Command-line overrides take precedence over YAML.

## Pipeline flow

```text
hotel.db/bookings
  -> deterministic cleaning and target/status integrity check
  -> remove duplicate booking IDs
  -> booking-time feature engineering
  -> temporal split: 2023 train / 2024 validation / 2025 holdout
  -> training-only imputation, scaling, rare-category grouping, one-hot encoding
  -> configurable classifier
  -> choose F1 threshold on validation only
  -> evaluate once on 2025 and save model, metrics, config, predictions
```

This order prevents preprocessing statistics, category vocabularies, and threshold selection from seeing the holdout period.

## EDA summary

The raw database contains 118,549 booking rows and 17,583 review rows. Initial checks found all SQLite booking columns stored as text, mixed boolean spellings, month abbreviations, whitespace/case variants and typos in categories, 353 duplicate booking IDs, and impossible or extreme values such as negative lead time. The raw outcome distribution is 75,398 check-outs, 41,969 cancellations, and 1,182 no-shows.

**TODO after completing `eda.ipynb`:** add 4-6 evidence-backed findings, including hotel/time differences, strongest booking-time associations, drift between years, anomaly decisions, and what each finding changed in the pipeline. Keep charts and detailed analysis in the notebook.

## Feature processing and leakage contract

| Feature group | Processing | Rationale |
|---|---|---|
| Target | Normalise `1/0`, `true/false`, `Yes/No`; cross-check against reservation status | Fix inconsistent text encoding and detect label contradictions |
| Dates | Construct arrival date from year/month/day; derive month, weekday, weekend arrival | Support temporal split and seasonality without using outcome dates |
| Counts/rates | Parse numeric; median-impute; robust-scale; add total nights/guests and log ADR | Preserve information while reducing sensitivity to malformed/extreme values |
| Categories | Trim/collapse whitespace, normalise case, map common typos; impute; group rare levels; one-hot encode | Handle dirty text and unseen future categories |
| Duplicate IDs | Keep one row per `booking_id` and report count removed | Avoid double-counting and cross-split contamination |
| High-cardinality agency/company | Retain as categories with rare-level grouping | They may carry channel/account signal without creating one column per rare ID |

The following are deliberately excluded:

- `reservation_status`, `reservation_status_date`: directly reveal the outcome.
- `assigned_room_type`: assigned at check-in, after the prediction cutoff.
- `booking_changes`: accumulates changes after booking creation; its value at confirmation cannot be reconstructed.
- `client_email`, `booking_id`: PII/identifiers with near one-to-one memorisation risk.
- `reviews.rating`, `reviews.comments`: only generated after completed stays, and absent for bookings that never stay.

`days_in_waiting_list` is retained because the brief defines it as elapsed before confirmation; `lead_time`, reserved room, ADR, deposit, promo and channel fields are assumed known at confirmation. These availability assumptions should be validated with the Reservations team before deployment.

## Models

Regularised logistic regression was the transparent linear baseline. Random forest was selected as the default because, on the 2024 validation set, it improved the declared primary/operational measures: average precision (0.737 versus 0.733), recall in the top 20% risk band (0.470 versus 0.456), and Brier score (0.169 versus 0.190). Logistic regression retained a higher validation ROC-AUC (0.786 versus 0.766), so the choice is a trade-off rather than universal dominance.

The random forest uses 300 trees and a minimum of five training rows per leaf. This is deliberately modest regularisation, not an exhaustive search. The prevalence-only dummy remains the minimum benchmark. Do not claim causal explanations from tree importance or interpretability from raw logistic coefficients without checking transformed feature names, correlated variables, and stability.

## Evaluation design and metrics

- **Temporal validation:** train on 2023, choose model/threshold on 2024, evaluate once on 2025. This better represents future use than a random split and exposes time drift.
- **Average precision:** primary ranking metric because it focuses on positive-case retrieval and is sensitive to class prevalence.
- **Recall at top 20%:** operational metric answering, "If staff can act on only the riskiest fifth of bookings, what fraction of fallout is found?"
- **ROC-AUC:** threshold-free discrimination summary, reported alongside rather than instead of average precision.
- **Brier score:** checks whether predicted probabilities are numerically useful for expected-value decisions.
- **Precision/recall/F1 and confusion matrix:** reported at a threshold selected on 2024 only. The current threshold maximises F1 as a neutral placeholder because intervention costs are unknown.

The table below is from the verified default run. Re-run it in the final environment and update only if the code/configuration changes. The 2025 fall in ROC-AUC and recall@capacity, alongside its higher 38.7% target prevalence, is evidence of temporal drift; 2025 must not be used for further tuning.

| Split/model | ROC-AUC | Average precision | Brier | Recall@top 20% | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| 2024 validation - random forest | 0.766 | 0.737 | 0.169 | 0.470 | 0.717 | 0.546 |
| 2025 holdout - random forest | 0.739 | 0.717 | 0.205 | 0.416 | 0.758 | 0.440 |
| 2025 prevalence-only dummy | 0.500 | 0.387 | 0.238 | 0.198 | 0.387 | 1.000 |

The F1-maximising validation threshold is 0.274. At that frozen threshold, the 2025 model trades lower recall for much higher precision than the dummy. The dummy predicts every row positive at this threshold, which explains its recall of 1.000 and makes clear why F1 alone is insufficient.

## Operational use and deployment considerations

- Replace the F1 threshold with an expected-value or capacity threshold after estimating intervention cost, saved-room value, contact capacity, and possible customer harm.
- Start with a human-reviewed pilot: reminders/reconfirmation are safer than automatically refusing or overbooking guests.
- Evaluate calibration and errors separately for City/Airport hotel, booking channels, lead-time bands, and seasons; monitor for disparate or proxy effects from geography/channel fields.
- Validate that every feature is available and has the same definition at scoring time. Log schema checks and fail closed on target/outcome columns.
- Monitor input drift, target prevalence, calibration, recall@capacity, customer complaints, and realised incremental room revenue. Retrain only after a documented trigger.
- Do not store or expose `client_email` in the model artefact or analyst prediction export; join risk scores to operational systems under access control.
- A model association is not causal. Use a controlled pilot to estimate whether interventions actually reduce fallout.

## Known limitations and next work

- The three yearly cohorts provide only two development periods and one holdout, so uncertainty and drift estimates are limited.
- The combined target serves a shared capacity outcome, but cancellation and no-show may require different interventions. A later hierarchical/two-stage model is worth testing if more data is available.
- The database lacks explicit booking-created timestamp, live inventory/occupancy, intervention history, competitor rates, and per-booking profit/cost.
- Category typo mappings are based on observed EDA and should become governed reference data in production.
- TODO: add evidence-led next steps from error analysis, calibration plots, subgroup results, and pilot design.

## Assessment documentation

Complete `decision_log.md` in your own words. In `prompt_chat_history.md`, include share links where supported and raw unedited transcripts, including this project's AI-assisted work. See `SUBMISSION_CHECKLIST.md` before submission.
