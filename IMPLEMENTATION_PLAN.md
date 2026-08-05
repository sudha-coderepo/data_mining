# Full Implementation Plan
## Sentiment & Theme Classification of Amazon Product Reviews (Updated)

This plan replaces the circular LLM-as-ground-truth evaluation with a **hybrid validation strategy**:

1. **Primary gold set** — 100–200 human-labeled reviews from your dataset (never used in training)
2. **Secondary validation** — OATS-ABSA (optional stretch — skip if behind)
3. **Proxy validation** — star-rating sentiment (already available)
4. **Distillation track** — LLM mimicry metrics (internal only, not reported as real-world accuracy)

**Audience:** Graduate data mining project team  
**Dataset:** 1,177 Amazon device reviews (`cleaned_amazon_reviews.csv`)  
**Timeline:** **2 weeks total** — Week 1 done, **you are in Week 2 now**  
**Repo baseline:** [sudha-coderepo/data_mining](https://github.com/sudha-coderepo/data_mining)

---

## Sprint Status (2-Week Plan)

| Week | Status | What should be done |
|------|--------|---------------------|
| **Week 1** | ✅ Done (assumed) | Data cleaned, LLM labels on all 1,177 reviews, existing models trained, `Report.md` draft |
| **Week 2** | 🔴 **NOW** | Human gold labels → retrain/evaluate → update report & slides → demo |

### Scope cuts for 2 weeks (read this first)

| Item | Full plan | **2-week plan** |
|------|-----------|-----------------|
| Human gold set | 200 reviews | **100 minimum** (200 if 2 people label 50 each in parallel) |
| OATS external validation | Required | **Optional stretch** — skip if behind |
| Hyperparameter tuning | GridSearch on val | **Skip** — use defaults + `class_weight='balanced'` |
| 5-fold cross-validation | Required | **Skip** — single split only |
| New folder structure | Full reorg | **Minimal** — add `splits/`, `labeling/`, `outputs/` only |
| Feature ablation | Word + char TF-IDF | **Skip** — keep char TF-IDF only |
| LLM self-consistency | 10% sample | **Skip** — already have LLM vs stars Kappa |

### Week 2 day-by-day schedule

| Day | Focus | Hours (team) | Deliverable |
|-----|-------|--------------|-------------|
| **Day 1 (Mon)** | Splits + export labeling sheet | 2–3h | `gold_labeling_template.csv`, start labeling |
| **Day 2 (Tue)** | Finish human labeling | 4–6h | `reviews_human_gold.csv` (100–200 rows) |
| **Day 3 (Wed)** | Preprocess + train + evaluate | 3–4h | Models retrained, `track_a_gold.csv`, figures |
| **Day 4 (Thu)** | Report + slides rewrite | 3–4h | Updated `Report.md`, `slides_guide.md` |
| **Day 5 (Fri)** | Demo + rehearsal + buffer | 2–3h | `demo.ipynb` works, presentation ready |

**Parallel work if 2 team members:**
- Person A: Label 50–100 reviews + write report methods/results
- Person B: Label 50–100 reviews + build `evaluation.py` + figures
- Both: Day 3 train/eval together; Day 4 slides; Day 5 rehearsal

---

## Table of Contents

1. [Goals & Success Criteria](#1-goals--success-criteria)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Project Structure](#3-project-structure)
4. [Phase 0 — Setup & Scope](#phase-0--setup--scope-week-1--done)
5. [Phase 1 — Data & Splits](#phase-1--data--splits-week-2-day-1)
6. [Phase 2 — LLM Silver Labels](#phase-2--llm-silver-labels-week-1--done)
7. [Phase 3 — Human Gold Set](#phase-3--human-gold-set-week-2-days-1-2)
8. [Phase 4 — Preprocessing & Features](#phase-4--preprocessing--features-week-2-day-3)
9. [Phase 5 — Model Training](#phase-5--model-training-week-2-day-3)
10. [Phase 6 — Evaluation Framework](#phase-6--evaluation-framework-week-2-day-3)
11. [Phase 7 — External Validation (OATS)](#phase-7--external-validation-oats-week-2-stretch)
12. [Phase 8 — Demo, Report & Presentation](#phase-8--demo-report--presentation-week-2-days-4-5)
13. [Evaluation Tracks Reference](#evaluation-tracks-reference)
14. [Labeling Guide (Appendix A)](#appendix-a-human-labeling-guide)
15. [Known Fixes to Current Repo](#known-fixes-to-current-repo)
16. [Deliverables Checklist](#deliverables-checklist)

---

## 1. Goals & Success Criteria

### Business goal
Automatically route Amazon product reviews by **sentiment** and **complaint theme** so support, product, and logistics teams do not manually triage thousands of reviews.

### Technical goals

| Goal | Target | How measured |
|------|--------|--------------|
| Sentiment accuracy on human gold | Macro F1 ≥ 0.50 | Track A (100+ human-labeled reviews) |
| Theme routing accuracy on human gold | Macro F1 ≥ 0.40 | Track A (rare classes are hard) |
| Beat naive baselines | +10 pts Macro F1 vs majority class | Track A baselines |
| LLM label quality | Cohen's Kappa ≥ 0.45 vs stars (sentiment) | Already ~0.46 in current repo |
| Human–LLM theme agreement | Report Kappa on gold overlap | Track A |
| Production viability | Inference < 10 ms/review locally | `demo.py` benchmark |
| Honest reporting | Primary metrics from human gold, not LLM | Report structure |

### Non-goals (explicitly out of scope)
- Multi-label classification (one theme per review only, for v1)
- Real-time API deployment
- Multilingual translation pipeline

---

## 2. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA                                                           │
│  Amazon Review Dataset.csv → data_exploration.py → cleaned (1,177 rows) │
│  → create_splits.py → train / val / gold_holdout (200, never train)     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌──────────────────┐   ┌──────────────────────┐   ┌─────────────────────┐
│ PHASE 2: LLM     │   │ PHASE 3: HUMAN GOLD  │   │ PHASE 7: OATS       │
│ Silver labels    │   │ 200 reviews labeled  │   │ External validation │
│ (train+val only) │   │ (evaluation only)    │   │ (mapped labels)     │
└──────────────────┘   └──────────────────────┘   └─────────────────────┘
          │                         │                         │
          └─────────────────────────┼─────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: PREPROCESSING → char TF-IDF (+ optional word TF-IDF ablation)  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 5: TRAINING (train+val only, class_weight='balanced')             │
│  Sentiment target: sentiment_rating (stars) OR llm_sentiment (ablation)│
│  Theme target:     llm_category (silver labels)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 6: EVALUATION                                                     │
│  Track A (PRIMARY):  vs human gold — report in slides & Report.md       │
│  Track B (SECONDARY): vs star ratings — sentiment sanity check          │
│  Track C (INTERNAL): vs LLM labels — distillation quality only          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 8: demo.py + updated Report.md + slides                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Project Structure

Create this layout inside `data_mining/`:

```
data_mining/
├── IMPLEMENTATION_PLAN.md          ← this file
├── data/
│   ├── cleaned_amazon_reviews.csv  ← existing (symlink or copy)
│   ├── reviews_labeled_llm.csv     ← existing
│   ├── reviews_human_gold.csv      ← NEW: 200 human-labeled reviews
│   └── external/oats_mapped.csv    ← NEW: OATS subset with mapped labels
├── splits/
│   ├── train_indices.pkl           ← NEW: canonical split
│   ├── val_indices.pkl
│   └── gold_indices.pkl            ← 200 IDs reserved for human labeling
├── models/                         ← NEW: all .pkl models go here
├── outputs/
│   ├── metrics/                    ← CSV metric tables per track
│   ├── figures/                    ← PNG confusion matrices & comparisons
│   └── error_analysis/             ← misclassified review samples
├── scripts/                        ← NEW: refactored pipeline scripts
│   ├── create_splits.py
│   ├── sample_gold_set.py
│   ├── merge_human_labels.py
│   ├── preprocessing.py            ← moved/refactored from root
│   ├── training.py
│   ├── evaluation.py               ← replaces comparison.py
│   ├── baselines.py
│   ├── external_oats_validation.py
│   └── demo.py
├── labeling/
│   ├── gold_labeling_template.csv  ← export for Google Sheets
│   └── LABELING_GUIDE.md           ← annotator instructions
├── data_exploration.py             ← keep at root or move to scripts/
├── llm_labeling.py
├── demo.ipynb
├── Report.md
└── requirements.txt
```

---

## Phase 0 — Setup & Scope (Week 1 — DONE)

**Status:** ✅ Should already be complete from Week 1.

### Week 1 checklist (verify today if anything missing)

- [x] Repo pulled; `cleaned_amazon_reviews.csv` has 1,177 rows
- [x] `reviews_labeled_llm.csv` complete (all reviews LLM-labeled)
- [x] Python env + `pip install -r requirements.txt`
- [x] Existing pipeline ran: `preprocessing.py`, `training.py`, `comparison.py`
- [ ] **If not done yet:** `pip install openpyxl` (for Excel labeling export)

### Decision locked for Week 2

- **Sentiment training target:** `sentiment` from star ratings (aligns train & eval)
- **Theme training target:** `llm_category` (silver labels — no time to relabel 977 rows)
- **Gold holdout:** 100 reviews minimum, 200 if both members label in parallel

---

## Phase 1 — Data & Splits (Week 2, Day 1)

**Duration:** 1–2 hours  
**Script to create:** `scripts/create_splits.py`  
**When:** Monday morning (first task of Week 2)

### Tasks

1. **Load** `reviews_labeled_llm.csv` (1,177 rows — has LLM columns)
2. **Reserve gold holdout first** — 100 or 200 IDs stratified by `llm_category`
3. **Split remaining rows:** 80% train, 20% val, stratified on `llm_category`
4. **Save indices once** — all scripts read from `splits/*.pkl`

### Stratified gold sample targets

**100-review minimum (1 person, ~3 hours labeling):**

| LLM category | Count |
|--------------|-------|
| Delivery issue | 12 |
| Product quality issue | 12 |
| Price complaint | 10 |
| Customer service issue | 10 |
| Feature request | 10 |
| Other | 20 |
| Star ≠ LLM sentiment (prioritize in sample) | 26 overlay |

**200-review stretch (2 people × 50 each, ~3 hours each):** double all counts above.

### `create_splits.py` requirements

```python
# Pseudocode — implement in scripts/create_splits.py
GOLD_SIZE = 200
RANDOM_STATE = 42

# Step 1: stratified sample → gold_indices.pkl (NEVER in train/val/test)
# Step 2: remaining → train_indices.pkl + val_indices.pkl
# Step 3: write splits/split_summary.json with class counts per split
```

### Fix from current repo
- **Remove** duplicate `train_test_split` calls in `training.py`
- **Remove** mismatched `test_indices.pkl` (category-stratified only)
- Single source of truth: `splits/train_indices.pkl`, `splits/val_indices.pkl`, `splits/gold_indices.pkl`

### Deliverables
- `splits/train_indices.pkl`, `splits/val_indices.pkl`, `splits/gold_indices.pkl`
- `splits/split_summary.json`

---

## Phase 2 — LLM Silver Labels (Week 1 — DONE)

**Status:** ✅ Already complete in repo (`reviews_labeled_llm.csv`, 1,177 rows).

### Already reported (use in report)
- Sentiment agreement vs stars: **82.24% accuracy, Kappa 0.4631**
- LLM labeling cost: **~$0.13**

### Week 2 action
- **Do not re-run** unless labels are missing — save time
- Cite existing LLM validation numbers in `Report.md` Section 3

---

## Phase 3 — Human Gold Set (Week 2, Days 1–2)

**Duration:** 1–2 days — **CRITICAL PATH**  
**Scripts:** `scripts/create_splits.py`, `scripts/validate_human_labels.py`, `scripts/merge_human_labels.py`  
**When:** Monday export + start; **finish by Tuesday EOD**

### Automated workflow (already built)

```bash
# Step A — Agent ran this (splits + template export)
python scripts/create_splits.py

# Step B — YOU label in labeling/gold_labeling_template.csv
#          Save as labeling/gold_labeling_completed.csv

# Step C — Validate before merge (required)
python scripts/validate_human_labels.py labeling/gold_labeling_completed.csv

# Step D — Merge into gold dataset (only if validation passed)
python scripts/merge_human_labels.py labeling/gold_labeling_completed.csv

# Step E — Tell agent "Human labels are ready" → training + Track A eval
python scripts/run_post_labeling_pipeline.py
```

See `labeling/README.md` for full instructions.

### Step 3.1 — Export labeling template

`scripts/sample_gold_set.py` should:
1. Load `splits/gold_indices.pkl`
2. Export `labeling/gold_labeling_template.csv` with columns:

| Column | Description |
|--------|-------------|
| `id` | Review ID |
| `reviews.text` | Full review text |
| `reviews.title` | Title |
| `reviews.rating` | 1–5 stars |
| `sentiment_rating` | Auto-filled: mapped from stars |
| `llm_sentiment` | Reference only (do not copy blindly) |
| `llm_category` | Reference only |
| `human_sentiment` | **BLANK — annotator fills** |
| `human_category` | **BLANK — annotator fills** |
| `annotator` | Name |
| `notes` | Optional disagreement notes |

Also export `.xlsx` for Google Sheets / Excel.

### Step 3.2 — Annotate (fast protocol)

**Speed tips:** ~2 min/review → 100 reviews ≈ 3–4 hours total.

- **1 person:** Label 100 reviews (minimum viable)
- **2 people:** Each labels 50–100 → 100–200 total; overlap 25 for Kappa
- Hide `llm_sentiment` and `llm_category` columns while labeling (or delete from sheet)
- Use Appendix A definitions; flag hard cases in `notes`, don't stall
- **Rule:** Label from **text**, not stars or LLM

### Step 3.3 — Merge & validate

`scripts/merge_human_labels.py`:
1. Load completed template
2. Validate allowed label values
3. Compute Cohen's Kappa (annotator 1 vs 2 on overlap of 50)
4. Adjudicate disagreements (discuss or third pass)
5. Save `data/reviews_human_gold.csv`

### Quality gates (relaxed for 2-week sprint)

| Check | Minimum acceptable |
|-------|-------------------|
| Human sentiment Kappa (2 annotators, n=25 overlap) | ≥ 0.50 (report whatever you get) |
| Human theme Kappa (2 annotators, n=25 overlap) | ≥ 0.40 |
| Blank labels in final gold set | 0 |
| Gold IDs overlap with train/val | 0 |
| Total gold labels | **≥ 100** |

### Deliverables
- `labeling/gold_labeling_template.csv` (blank)
- `data/reviews_human_gold.csv` (completed)
- `outputs/human_agreement_report.txt`

---

## Phase 4 — Preprocessing & Features (Week 2, Day 3)

**Duration:** 1–2 hours  
**Script:** update existing `preprocessing.py` (minimal changes)

### Tasks

1. **Merge** LLM labels into preprocessed dataframe (train+val only for fitting vectorizer)
2. **Text field:** `combined_text = reviews.title + " " + reviews.text`
3. **Cleaning:** lowercase, strip HTML, preserve international chars (match existing approach)
4. **Primary features:** char TF-IDF, `ngram_range=(2,4)`, `max_features=5000`
5. **Critical:** Fit vectorizer on **train indices only** (prevent leakage)
6. ~~Word TF-IDF ablation~~ — skip for 2-week sprint

### Outputs

```python
# Fit on train only
vectorizer.fit(df.loc[train_idx, 'cleaned_text'])
X = vectorizer.transform(df.loc[all_non_gold_idx, 'cleaned_text'])
```

### Deliverables
- `data/preprocessed_reviews.csv`
- `models/tfidf_vectorizer.pkl`
- `models/tfidf_matrix_trainval.pkl`
- `outputs/feature_ablation_notes.md` (optional)

---

## Phase 5 — Model Training (Week 2, Day 3)

**Duration:** 1–2 hours  
**Script:** update existing `training.py`

### Models to train

| Model | sklearn class | Notes |
|-------|---------------|-------|
| Naive Bayes | `MultinomialNB()` | Fast baseline |
| Logistic Regression | `LogisticRegression(max_iter=1000, class_weight='balanced')` | Strong linear baseline |
| Random Forest | `RandomForestClassifier(n_estimators=100, class_weight='balanced')` | Production candidate |

### Training configuration

```python
# Sentiment — recommended primary
y_sentiment = df['sentiment']  # from star ratings

# Theme — silver labels (no human theme labels in train for v1)
y_theme = df['llm_category']

# Always use splits/train_indices.pkl for fit
# Use splits/val_indices.pkl for hyperparameter tuning
# NEVER touch splits/gold_indices.pkl
```

### Hyperparameter tuning

**Skip for 2-week sprint.** Use defaults + `class_weight='balanced'`. Existing `.pkl` models can be reused after fixing splits if time is very tight.

### Deliverables
- `models/sentiment_{nb,lr,rf}.pkl`
- `models/category_{nb,lr,rf}.pkl`
- `outputs/training_log.txt`

---

## Phase 6 — Evaluation Framework (Week 2, Day 3)

**Duration:** 2–3 hours  
**Scripts:** update `comparison.py` → or new `evaluation.py`, plus `baselines.py`

### Three evaluation tracks

#### Track A — PRIMARY (report these numbers)

**Data:** `data/reviews_human_gold.csv` (100–200 reviews)  
**Ground truth:** `human_sentiment`, `human_category`

Report for each model + baselines:
- Accuracy
- Macro Precision, Recall, F1
- Weighted F1
- Per-class precision/recall table
- Confusion matrix → `outputs/figures/`

#### Track B — SECONDARY (sentiment sanity check)

**Data:** validation set (195 reviews)  
**Ground truth:** `sentiment` (star mapping)  
**Purpose:** Compare with original repo results; show star-proxy behavior

#### Track C — INTERNAL (appendix only)

**Data:** validation set  
**Ground truth:** `llm_sentiment`, `llm_category`  
**Purpose:** Measure distillation — "how well does RF mimic Gemini?"  
**Do NOT present as real-world accuracy**

### Baselines (required in Track A)

`scripts/baselines.py` implements:

| Baseline | Sentiment | Theme |
|----------|-----------|-------|
| Majority class | Always `Positive` | Always `Other` |
| Star rating mapping | Use `sentiment_rating` | N/A |
| Keyword rules | N/A | Regex keyword lists (see below) |
| LLM zero-shot | `llm_sentiment` | `llm_category` |
| Best ML model | RF or tuned LR | RF |

**Keyword rules example (theme baseline):**

```python
RULES = {
    "Delivery issue": ["shipping", "delivery", "arrived", "package", "late", "delayed"],
    "Product quality issue": ["broken", "defect", "quality", "stopped working", "battery"],
    "Price complaint": ["expensive", "overpriced", "price", "worth", "money"],
    "Customer service issue": ["support", "return", "refund", "service", "warranty"],
    "Feature request": ["wish", "would be nice", "should add", "feature", "update"],
}
# Default → Other
```

### Statistical reporting (2-week minimum)

- ~~5-fold CV~~ — skip
- ~~Bootstrap CI~~ — skip unless time on Day 5
- Error analysis: export **10** misclassified examples per task (quick qualitative slide)

### Deliverables
- `outputs/metrics/track_a_gold.csv`
- `outputs/metrics/track_b_stars.csv`
- `outputs/metrics/track_c_distillation.csv`
- `outputs/figures/sentiment_confusion_gold.png`
- `outputs/figures/category_confusion_gold.png`
- `outputs/error_analysis/sentiment_errors.csv`
- `outputs/error_analysis/category_errors.csv`

---

## Phase 7 — External Validation (OATS) (Week 2 — STRETCH)

**Duration:** 2–3 hours  
**Priority:** P2 — **only if Days 1–4 are on track**  
**Script:** `scripts/external_oats_validation.py`

### Why
Shows evaluators you did not only test on self-created labels.

### Steps

1. Load OATS-ABSA from Hugging Face:

```python
from datasets import load_dataset
ds = load_dataset("jordiclive/OATS-ABSA")
# Filter amazon_ff domain
```

2. Map OATS aspect categories → your 6 theme labels (see mapping table below)
3. Sample 200 reviews with diverse aspects
4. Run your trained RF models (no retraining on OATS)
5. Report Macro F1 with caveat: **domain shift** (Fine Food ≠ Amazon Devices)

### OATS → project label mapping

| OATS / aspect keyword | Your category |
|-----------------------|---------------|
| delivery, shipping, packaging | Delivery issue |
| quality, taste, freshness, portion | Product quality issue |
| price, value | Price complaint |
| service, support | Customer service issue |
| suggestion, wish (if present) | Feature request |
| general praise, no specific complaint | Other |

### Deliverables
- `data/external/oats_mapped.csv`
- `outputs/metrics/track_external_oats.csv`
- Short paragraph in Report.md explaining domain limitation

---

## Phase 8 — Demo, Report & Presentation (Week 2, Days 4–5)

**Duration:** 1.5 days  
**When:** Thursday = report + slides; Friday = demo + rehearsal

### 8.1 — Interactive demo (`scripts/demo.py`)

```bash
python scripts/demo.py --text "Great product but shipping took two weeks"
# Output:
#   Sentiment: Negative (confidence 0.72)
#   Theme: Delivery issue (confidence 0.81)
#   Flag: low confidence → suggest human review
```

### 8.2 — Update `Report.md`

Restructure with these sections:

1. Problem & business case
2. Dataset (1,177 reviews + 200 human gold + OATS external)
3. Methodology (3 label layers: stars, LLM silver, human gold)
4. **Results — Track A primary table (human gold)**
5. Results — Track B & C in appendix
6. External validation (OATS)
7. Error analysis & limitations
8. Production recommendation (RF local deployment)
9. Future work (multi-label, larger dataset, human-in-the-loop)

### 8.3 — Update slides (`slides_guide.md`)

Key narrative change:

> "We validated theme classification against **200 human-labeled reviews**, not LLM self-evaluation. Random Forest achieved **X% Macro F1** on human gold, beating keyword baselines by **Y points**."

### 8.4 — Model card (`outputs/MODEL_CARD.md`)

Document: training data, label definitions, gold F1, known failure modes, confidence thresholds.

### Deliverables
- `scripts/demo.py`
- Updated `Report.md`
- Updated `slides_guide.md`
- `outputs/MODEL_CARD.md`
- `demo.ipynb` refreshed

---

## Evaluation Tracks Reference

| Track | Split | Ground truth | Use in report |
|-------|-------|--------------|---------------|
| **A — Gold** | 200 human-labeled | `human_sentiment`, `human_category` | **Main results table** |
| **B — Stars** | Val set (195) | Star-mapped `sentiment` | Secondary / sentiment sanity |
| **C — Distillation** | Val set (195) | `llm_sentiment`, `llm_category` | Appendix only |
| **External** | OATS 200 sample | Mapped human OATS labels | Discussion / robustness |

---

## Appendix A: Human Labeling Guide

Copy to `labeling/LABELING_GUIDE.md`.

### Sentiment (label from text, not stars)

| Label | Definition | Example |
|-------|------------|---------|
| **Positive** | Clear satisfaction, recommendation | "Love my Kindle, best purchase ever" |
| **Neutral** | Mixed or factual, no strong emotion | "It works as expected, nothing special" |
| **Negative** | Dissatisfaction, frustration, regret | "Stopped working after a week" |

**Important:** A 5-star review can be Neutral or Negative if the text is lukewarm or critical.

### Theme (pick exactly ONE primary category)

| Label | Definition | Example |
|-------|------------|---------|
| **Delivery issue** | Shipping, packaging, arrival condition | "Package arrived crushed" |
| **Product quality issue** | Defects, durability, performance | "Screen flickers constantly" |
| **Price complaint** | Cost, value for money | "Too expensive for what you get" |
| **Customer service issue** | Returns, refunds, support experience | "Amazon support was no help" |
| **Feature request** | Suggestions, missing features | "Wish it had a backlight toggle" |
| **Other** | General praise, off-topic, no actionable theme | "Great gift for my mom" |

**Tie-break rule:** If two themes apply, pick the one that would **route to the most urgent team**.

### Annotator checklist
- [ ] Read full review text before labeling
- [ ] Do not look at LLM columns until after labeling (hide in spreadsheet)
- [ ] Flag ambiguous cases in `notes` column
- [ ] Take breaks every 50 reviews

---

## Known Fixes to Current Repo

| Issue | Current behavior | Fix |
|-------|------------------|-----|
| Split inconsistency | `training.py` splits on sentiment AND category separately; `test_indices.pkl` uses category only | Single `create_splits.py` |
| Circular theme eval | `comparison.py` evaluates vs `llm_category` | Track A uses `human_category` |
| Sentiment train/eval mismatch | Train on `llm_sentiment`, eval vs stars | Train on `sentiment` (stars) for main model |
| No baselines | Only 3 ML models | Add majority class + keyword rules |
| Class imbalance ignored | Default sklearn settings | `class_weight='balanced'` |
| Vectorizer leakage | Fit on all data | Fit on train indices only |
| Models in repo root | `.pkl` files scattered | Move to `models/` |

---

## Deliverables Checklist

**Priority:** P0 = must ship Week 2 · P1 = should have · P2 = stretch

### Code
- [x] P0 `scripts/create_splits.py` (splits + template export)
- [x] P0 `scripts/validate_human_labels.py` (run when labels returned)
- [x] P0 `scripts/merge_human_labels.py` (merge after validation passes)
- [ ] P0 Update `training.py` (use splits, exclude gold, `class_weight='balanced'`)
- [ ] P0 Update `comparison.py` or `evaluation.py` (Track A on human gold)
- [ ] P1 `scripts/baselines.py` (majority class + keyword rules)
- [ ] P1 Refresh `demo.ipynb`
- [ ] P2 `scripts/external_oats_validation.py`

### Data
- [x] P0 `splits/gold_indices.pkl` (+ train/val)
- [x] P0 `labeling/gold_labeling_template.csv` (100 reviews to label)
- [ ] P0 `data/reviews_human_gold.csv` (**≥ 100 labeled** — you fill this)
- [ ] P2 `data/external/oats_mapped.csv`

### Outputs
- [ ] P0 `outputs/metrics/track_a_gold.csv`
- [ ] P0 `outputs/figures/` (2 confusion matrices + 1 comparison chart)
- [ ] P1 `outputs/error_analysis/` (10 examples per task)
- [ ] P2 `outputs/MODEL_CARD.md`

### Documentation
- [ ] P0 Updated `Report.md` (human gold as primary eval)
- [ ] P0 Updated `slides_guide.md`
- [ ] P1 `labeling/LABELING_GUIDE.md` (copy from Appendix A)

---

## 2-Week Timeline Summary

| Week | Phases | Key milestone |
|------|--------|---------------|
| **1** ✅ | 0, 2 + existing pipeline | Data clean, LLM labels, initial models & `Report.md` draft |
| **2** 🔴 | 1, 3, 4, 5, 6, 8 (+7 stretch) | Human gold → retrain/eval → report → present |

### Week 2 at a glance

```
Mon     Tue         Wed              Thu           Fri
────    ────        ───              ───           ───
Splits  Finish      Preprocess       Report.md     demo.ipynb
Export  human       Train            slides        Rehearsal
label   gold 100+   Evaluate Track A  Error ex.    Buffer
sheet   labels      Figures          slides_guide
```

---

## Requirements Additions

Add to `requirements.txt`:

```
datasets
huggingface_hub
openpyxl
```

---

## Presentation of Results (Template)

### Primary table (Track A — Human Gold, n=100–200)

| Model | Sentiment Macro F1 | Theme Macro F1 |
|-------|-------------------|----------------|
| Majority class | — | — |
| Keyword rules | N/A | — |
| Naive Bayes | — | — |
| Logistic Regression | — | — |
| **Random Forest** | — | — |
| LLM zero-shot | — | — |

Fill after Phase 6.

### Honest limitations slide (required)

- 100–200 gold labels from same dataset (not independent collection)
- Theme training still uses LLM silver labels (gold eval only)
- Single product domain (Amazon devices)
- English only
- OATS external validation has domain shift

---

*Last updated: 2026-08-02 — **2-week sprint** (Week 1 done, Week 2 in progress). Human gold primary; OATS optional stretch.*
