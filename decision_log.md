# AIAP Technical Assessment — Decision Log

**Candidate name (as in NRIC):** Singaravelan

**Email (as used in your application):** singavln32@gmail.com

---

## A note on this document

This decision log is the primary instrument by which we understand your thinking. The questions below cover the reasoning behind your work — from how you define the problem at the start to the decisions you made during the work itself. Do answer all five questions in your own words. 

You may use AI assistance freely on the technical deliverables (the EDA and the ML pipeline), but this Decision Log itself should be written by you — it is the record of your own thinking that we cross-check against your chat history.


---

## 1. Clarifying questions

What questions would you ask to better define and narrow the problem statement? For each question, briefly explain how the answer would meaningfully change your approach. 

Note: If it helps your decision-making, you may assume and list out the stakeholders' likely answers.


**Your answer:**

The first question I would ask is who will use the prediction and what action they can realistically take. A risk score is only useful if Revenue and Reservations can respond, for example by sending reminders, asking for reconfirmation, reviewing deposit terms, or using the score as one input to controlled overbooking. If the only permitted action were pricing, I would frame the problem differently and would need demand, inventory, competitor-price, and price-response data that are not available here.

Second, I would ask when the prediction must be available. I assumed the decision point is when a reservation is confirmed. This assumption determines which fields are valid predictors. At that time, lead time, arrival date, reserved room, ADR, deposit type, channel, promo code, and prior booking history should be known. In contrast, final reservation status, its date, the room actually assigned at check-in, later booking changes, and post-stay reviews are not available and must not be used. I would confirm the point-in-time meaning of `days_in_waiting_list`, `agent`, and `company` with the data owner.

Third, I would ask whether cancellations and no-shows lead to the same operational response. If they require very different actions, I would consider separate models or a two-stage model. For this assessment I assumed their immediate business consequence is shared: a reserved room fails to become an occupied stay. The data also contains only 1,182 no-shows compared with 41,969 cancellations, so a no-show-only model would have much less evidence.

Fourth, I would ask about intervention capacity and economics: how many bookings can staff contact, what an empty room costs, what a successful intervention is worth, and what customer harm or workload a false alarm creates. Those answers would determine the final operating threshold. Because they were unavailable, I used recall in the top 20% risk band as a capacity-style metric and selected a provisional F1 threshold on validation data. I would not treat that threshold as a production business rule.

Finally, I would ask how the records were generated and whether any previous intervention, cancellation-policy change, duplicated extraction, or system migration occurred from 2023 to 2025. This matters because the data has mixed encodings, 353 duplicated booking IDs, category spelling variants, impossible negative lead times, and a higher outcome prevalence in 2025. These observations make a chronological evaluation and explicit data-quality checks more defensible than a random split.

---

## 2. Defining the Problem Statement

Restate, in your own words, the refined problem you decided to solve. List your key assumptions. Briefly note what other framings you considered, and what you deliberately left out or scoped down, and why.


**Your answer:**

At reservation confirmation, predict the probability that each booking will fail to materialise as an occupied stay because it is cancelled or becomes a no-show, rather than ending in check-out. The output is a ranked list of booking-level probabilities for the Revenue and Reservations team. It is intended to prioritise limited, low-harm actions such as reminders and reconfirmation and to support, but not automate, carefully controlled overbooking decisions.

I define the binary target from `is_canceled`, after normalising its inconsistent representations and cross-checking it against `reservation_status`. The cross-check showed that positive values cover both `Canceled` and `No-Show`, while negative values correspond to `Check-Out`. Combining the outcomes is justified by their common empty-room consequence and by the small number of no-shows. The raw data contains 118,549 booking rows: 75,398 check-outs, 41,969 cancellations, and 1,182 no-shows.

My key assumptions are that the model is scored at confirmation time; retained predictors are available with their historical point-in-time values; the target represents the operational outcome of interest; and a risk ranking is useful even before the exact intervention economics are known. I also assume that chronological performance on 2025 is more representative of future deployment than a random holdout.

I considered three alternatives. A no-show-only model was too narrow and data-sparse. A three-class model could separate cancellation from no-show, but it would complicate evaluation and intervention design before the stakeholder has confirmed that the two outcomes require different actions. Price optimisation was attractive from the business statement but was scoped out because the database lacks live demand, room inventory, competitor prices, intervention history, and a credible causal price-response signal. Review sentiment was also excluded because reviews are written after completed stays and are structurally unavailable for bookings that cancel or do not show.

---

## 3. Key decisions during Solution Development

Walk through three key decisions you made during Solution Development. For each: what options did you consider, what did you choose, and why? These can be technical (modelling choices, feature handling, evaluation metrics) or about the work itself (what to prioritise, what to drop, how to spend your time).


**Your answer:**

**Decision 1 - prediction cutoff and leakage controls.** I considered using every column, predicting closer to arrival, or fixing the prediction at confirmation. I chose confirmation because it provides a clear and actionable cutoff and can be reconstructed most defensibly from the supplied data. I excluded `reservation_status` and `reservation_status_date` because they reveal the outcome, `assigned_room_type` because it is determined at check-in, and `booking_changes` because it accumulates after booking creation and I cannot reconstruct its value at confirmation. I also excluded `client_email` and `booking_id` to prevent PII or near-unique identifiers from becoming memorisation features. Reviews were not joined into the model because they occur after a stay. This choice sacrifices potentially predictive information in exchange for a model that can actually be scored at the stated decision time.

**Decision 2 - chronological development and holdout design.** I considered a random split, cross-validation across all years, and a year-based split. I used 2023 for training, 2024 for validation and model/threshold decisions, and 2025 as an untouched holdout. A random split would make development scores optimistic if category frequencies, policies, or customer behaviour change over time. The positive rate increased from about 35.0% in validation to 38.7% in the 2025 holdout, and both tested models lost some discrimination, which supports the concern about temporal drift. All imputation, category grouping, one-hot encoding, scaling, and model fitting occur inside a scikit-learn pipeline fitted on training data only.

**Decision 3 - model and metrics.** I compared regularised logistic regression with a random forest and used a prevalence-only dummy as the minimum benchmark. Logistic regression was the simpler linear baseline and achieved higher validation ROC-AUC (0.786 versus 0.766). I selected the random forest because it was better on the measures I had declared more relevant before evaluating the holdout: validation average precision (0.737 versus 0.733), recall in the top 20% risk band (0.470 versus 0.456), Brier score (0.169 versus 0.190), and F1 at the validation-selected threshold. On the untouched 2025 data, the selected model achieved ROC-AUC 0.739, average precision 0.717, Brier score 0.205, and recall@top-20% 0.416. The dummy average precision was 0.387. I retained ROC-AUC for broad ranking quality, average precision for positive-case retrieval, Brier score for probability quality, and recall@top-20% for a capacity-constrained operational interpretation. The weaker 2025 performance is a limitation and a reason to prioritise calibration and monitoring rather than further tune against the holdout.

---

## 4. Use of the AI assistant

Where did you use the AI assistant in this work? Give three specific examples of something the assistant suggested that you changed, rejected, or significantly modified — and explain your reasoning.


**Your answer:**

I used the AI assistant to extract and cross-check the assessment requirements, inspect the SQLite schema and data quality, propose the project structure, draft reusable pipeline code, and test the notebook and models. I treated its output as a starting point and made the following changes or qualifications.

First, the assistant initially proposed logistic regression as the default because it was simpler and interpretable. I did not keep that choice automatically. I required both model paths to run and compared them using 2024 validation evidence. I changed the default to random forest because it performed better on average precision, recall at the top-20% capacity band, Brier score, and F1, while documenting that logistic regression still had higher ROC-AUC. This prevented “more complex is better” or “simpler is always better” from replacing empirical selection.

Second, the assistant suggested choosing a threshold that maximised F1. I retained it only as a reproducible assessment placeholder. I rejected the idea that it is a defensible production threshold because F1 assigns an implicit and unverified trade-off between false positives and false negatives. A real threshold should use intervention capacity, saved-room value, customer impact, and false-alarm cost. I therefore emphasised threshold-free and capacity-based metrics and documented the threshold limitation.

Third, the assistant suggested retaining high-cardinality `agent` and `company` fields with rare-category grouping. I kept them provisionally because they may represent booking-channel or account behaviour, but I did not treat that as settled. These variables could overfit, drift, or act as proxies. I added unseen-category handling, temporal validation, data-drift monitoring, and subgroup/ablation analysis as necessary follow-up work. I also excluded direct identifiers and all post-outcome fields even though including them would have produced superficially stronger scores.

---

## 5. Next Steps

If you had one more week to continue this project, what would you do next, and why? What signals from your current work make those the right next steps?


**Your answer:**

With one more week, I would first perform a point-in-time feature audit with the Reservations data owner. The most important unresolved risk is whether each retained value is truly available and historically correct at confirmation. I would specifically verify waiting-list duration, agency/company fields, ADR changes, deposit changes, and whether any intervention or policy change occurred across the three years.

Second, I would complete calibration and temporal/subgroup diagnostics. The selected random forest's Brier score worsened from 0.169 on 2024 validation to 0.205 on 2025, while the positive rate increased. I would add reliability diagrams, calibration intercept/slope, bootstrap confidence intervals, and performance slices by hotel, lead-time band, market segment, channel, and season. I would compare Platt and isotonic calibration using development data only and leave 2025 untouched.

Third, I would convert the provisional threshold into a business decision rule. I would estimate contact capacity, intervention cost, room contribution margin, false-alarm harm, and the probability that an intervention changes the outcome. This would allow an expected-value or capacity threshold and a small human-reviewed pilot. The pilot should begin with low-harm reminders or reconfirmation and measure incremental outcomes against a control group because model association alone does not show that an intervention prevents fallout.

Finally, I would examine cancellation and no-show errors separately. If the features, timing, and useful actions differ, I would test a hierarchical approach: first predict booking fallout, then distinguish cancellation from no-show among high-risk bookings. I would also run ablations for `agent`, `company`, country, promo code, and potentially unstable categories, and formalise the observed typo mappings as governed reference data rather than permanent hand-written assumptions.

---
