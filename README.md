# hiver-sde-intern-assignment
Take‑home assignment for Hiver SDE Intern role.
# Hiver SDE Intern — Take-Home Assignment

## Golden Set Note
- Sampled ~200 Uber Support tweets.
- Labelled into 4 intents: Complaint, Information request, Refund issue, Escalation needed.

## Problem Framing
- Defined what “good” means for Uber Support: accurate intent detection, helpful replies, correct escalation.
- Chose not to build full conversation threading due to scope.

## Baselines
- Trivial baseline: TF-IDF + Logistic Regression.
- Improved baseline: SentenceTransformer embeddings + Logistic Regression.
- Balanced dataset with oversampling to handle class imbalance.

## Evaluation Harness
- Automated metrics: classification reports, confusion matrices.
- LLM-as-judge rubric: GPT scored replies on correctness, helpfulness, tone; compared with human ratings.

## Failure Analysis
- Top 5 failure modes with examples (sarcasm, multi-intent, ambiguity, slang, escalation boundary).

## Misleading Headline Number
- Explained why 0.97–1.0 accuracy is misleading (oversampling inflates scores, skewed distribution).

## Next Steps
- Collect more labelled data.
- Fine-tune transformer model.
- Add escalation logic with confidence thresholds.
- Deploy demo API.

## Decision Log
- 10–15 non-obvious decisions documented (brand choice, intent definition, oversampling, model selection, evaluation methods, etc.).

## How to Run
1. Clone repo.
2. Install requirements (`pip install -r requirements.txt`).
3. Run notebook end-to-end (takes <15 minutes).
