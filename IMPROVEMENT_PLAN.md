# Model Improvement Plan
## Sentiment & Theme Classification — Post Human-Gold Evaluation

**Audience:** Graduate data mining project team  
**Dataset:** 1,177 Amazon device reviews  
**Repo:** [sudha-coderepo/data_mining](https://github.com/sudha-coderepo/data_mining)  
**Status:** Planned — not yet implemented  
**Created:** August 2026  
**Related docs:** `IMPLEMENTATION_PLAN.md` (original 2-week sprint), `Report.md`, `dashboard.py`

---

## End-to-end pipeline (status map)

**Legend:** 🟢 Green = implemented · 🟡 Yellow = improvement plan (not yet built)

```mermaid
flowchart TB
    subgraph LEGEND[" "]
        direction LR
        LEG_DONE["🟢 Implemented"]
        LEG_TODO["🟡 Improvement plan"]
    end

    subgraph INGEST["1 · Data ingestion"]
        A1["Amazon Review Dataset<br/>1,177 reviews"]
        A2["Clean & preprocess text<br/>preprocessing.py"]
        A3["LLM silver labeling<br/>llm_labeling.py · Gemini"]
        A1 --> A2 --> A3
    end

    subgraph SPLIT["2 · Splits & human gold"]
        B1["Create splits<br/>861 train · 216 val · 100 gold<br/>create_splits.py"]
        B2["Human labeling<br/>50 + 50 annotators"]
        B3["Validate & merge labels<br/>validate / merge / combine scripts"]
        B4["reviews_human_gold.csv"]
        A3 --> B1
        B1 --> B2 --> B3 --> B4
    end

    subgraph FEAT["3 · Feature engineering"]
        C1["Char TF-IDF 2–4 · 5k features<br/>fit on train only<br/>preprocess_trainval.py"]
        C2["Word + char TF-IDF · title+body<br/>Phase 1.3 / 4.2"]
        C3["Star rating feature<br/>Phase 1.2"]
        B4 --> C1
        C2 -.-> C1
        C3 -.-> C1
    end

    subgraph TRAIN["4 · Model training"]
        D1["NB · LR · RF<br/>class_weight=balanced<br/>train_models.py"]
        D2["Sentiment target: star-mapped"]
        D3["Theme target: LLM labels"]
        D4["Train theme on human labels + CV<br/>Phase 1.1"]
        D5["LinearSVC · SGD · hyperparam tune<br/>Phase 2.1"]
        D6["Class merge / hierarchy<br/>Phase 2.2"]
        C1 --> D1
        D2 --> D1
        D3 --> D1
        D4 -.-> D1
        D5 -.-> D1
        D6 -.-> D1
    end

    subgraph EVAL["5 · Evaluation"]
        E1["Track A · human gold n=100<br/>Primary metric"]
        E2["Track B · val vs stars"]
        E3["Track C · val vs LLM distillation"]
        E4["Threshold tuning · Macro F1<br/>Phase 1.4"]
        E5["Metrics CSV + confusion PNGs<br/>evaluation.py"]
        D1 --> E1 & E2 & E3
        E1 & E2 & E3 --> E5
        E4 -.-> E5
    end

    subgraph AL["6 · Active learning loop"]
        F1["Score unlabeled pool<br/>uncertainty + disagreement<br/>select_labeling_candidates.py<br/>Phase 4"]
        F2["Export labeling batch<br/>25–50 reviews"]
        F3["Human label + validate + merge"]
        F4["Relabel guidelines · adjudication<br/>Phase 2.4"]
        F1 --> F2 --> F3
        F4 -.-> F3
        F3 -.->|retrain| D1
    end

    subgraph DEMO["7 · Demo & deployment"]
        G1["Streamlit dashboard<br/>Before · After · Live predict<br/>dashboard.py"]
        G2["Hybrid router<br/>ML → LLM → human<br/>Phase 3.2"]
        G3["Improved LLM prompts<br/>Phase 3.1"]
        G4["DistilBERT fine-tune<br/>Phase 3.3 optional"]
        E5 --> G1
        G2 -.-> G1
        G3 -.-> A3
        G4 -.-> D1
    end

    subgraph REPORT["8 · Reporting"]
        H1["Report.md · before/after §5.3"]
        H2["Slides guide · demo.ipynb"]
        H3["Report §5.5 ablation + AL rounds<br/>Phase 4.6"]
        E5 --> H1
        G1 --> H2
        H3 -.-> H1
    end

    classDef done fill:#22c55e,stroke:#15803d,stroke-width:2px,color:#052e16
    classDef todo fill:#eab308,stroke:#a16207,stroke-width:2px,color:#422006
    classDef neutral fill:#f1f5f9,stroke:#94a3b8,color:#334155

    class A1,A2,A3,B1,B2,B3,B4,C1,D1,D2,D3,E1,E2,E3,E5,G1,H1,H2 done
    class C2,C3,D4,D5,D6,E4,F1,F2,F3,F4,G2,G3,G4,H3 todo
    class LEG_DONE,LEG_TODO neutral
    style LEG_DONE fill:#22c55e,stroke:#15803d,color:#052e16
    style LEG_TODO fill:#eab308,stroke:#a16207,color:#422006
```

### Quick reference — implemented vs planned

| Stage | Implemented 🟢 | Planned 🟡 |
|-------|----------------|------------|
| **Data** | Clean text, LLM labels on 1,177 reviews | Improved LLM prompts (3.1) |
| **Labels** | 100 human gold, validate/merge scripts | Active learning batches (4.5), expand to 150–200 (2.3), tighter guidelines (2.4) |
| **Features** | Char TF-IDF, train-only fit, no leakage | Word+char, title+body (1.3/4.2), star rating (1.2) |
| **Training** | NB/LR/RF, balanced weights, star→sentiment, LLM→theme | Human-label theme target (1.1), SVC/SGD + tune (2.1), class hierarchy (2.2) |
| **Evaluation** | Track A/B/C, metrics CSV, figures | Threshold tuning (1.4), ablation tables (4.6) |
| **Demo** | Streamlit dashboard + live prediction | Hybrid ML→LLM→human router (3.2), optional BERT (3.3) |
| **Report** | §5.3 before/after, recommendations | §5.5 TF-IDF ablation + active learning rounds |

---

## Why scores look lower than before

The pipeline is **more honest**, not necessarily worse. Primary evaluation now uses **100 human-labeled reviews** (Track A) instead of LLM-as-ground-truth or star-only proxies.

| Task | Old setup (easier) | Current setup (harder) | Best on human gold |
|------|-------------------|------------------------|-------------------|
| **Sentiment** | ~86% acc on 236-test vs **stars** | 74% acc / **59% Macro F1** vs **human labels** | Logistic Regression |
| **Theme** | ~77% acc vs **LLM labels** (circular) | 56% acc / **44% Macro F1** vs **human labels** | LLM zero-shot |

### Constraints limiting headline metrics

1. **Human gold is a hard test set** — 49/100 gold reviews have star-mapped sentiment ≠ LLM sentiment.
2. **Theme train/eval mismatch** — models train on `llm_category` but Track A scores against `human_category` (~56% agreement).
3. **Class imbalance** — gold themes: Other (43), Product quality (26), Feature request (15), Price complaint (13), Customer service (3); Delivery barely represented.
4. **Macro F1 penalizes rare classes** — accuracy can look acceptable while Macro F1 stays low.

### Validation reference (Track B — not primary)

On 216 validation reviews vs star-mapped sentiment, Logistic Regression already reaches **~86% accuracy / ~69% Macro F1**. Use this as a sanity check; **Track A remains the primary metric for reporting**.

---

## Goals & success criteria

| Metric | Baseline (Track A, human gold) | Target (Phase 1–2) | Stretch |
|--------|--------------------------------|---------------------|---------|
| Sentiment Macro F1 | 59.0% (LR) | **65–72%** | 75%+ |
| Sentiment accuracy | 74% (LR) | **78–82%** | 85%+ |
| Theme Macro F1 | 43.7% (LLM) | **50–58%** | 65%+ |
| Theme accuracy | 56% (LLM) | **60–68%** | 70%+ |

**Reporting rule:** Always report **Macro F1 on human gold (Track A)** as primary. Track B/C are secondary. Do not chase 80%+ theme accuracy on human gold without label expansion or hybrid LLM routing — document why if targets are not met.

---

## Execution overview

```mermaid
flowchart TD
    P1[Phase 1: Quick wins] --> P2[Phase 2: Model and labels]
    P2 --> P4[Phase 4: TF-IDF and active learning]
    P2 --> P3[Phase 3: LLM hybrid stretch]
    P4 --> M[Rerun pipeline + update dashboard]
    P1 --> M
    P3 --> M
    M --> R[Update Report.md and slides]
```

| Phase | Effort | Expected impact |
|-------|--------|-----------------|
| **Phase 1** — Quick wins | 1–2 days | Highest ROI |
| **Phase 2** — Model & label quality | 3–5 days | Solid report improvements |
| **Phase 3** — LLM hybrid (stretch) | 2–3 days | Strong demo narrative |
| **Phase 4** — TF-IDF & active learning | 2–3 days | Smarter features + targeted labeling |

---

## Phase 1 — Quick wins (1–2 days)

### 1.1 Train theme models on human labels ⭐ Highest priority

**Problem:** Theme classifiers learn `llm_category` but are evaluated on `human_category`.

**Action:**
- [ ] Add training mode: target = `human_category` on gold-labeled rows
- [ ] Use **5-fold cross-validation** on the 100 human labels (80 train / 20 test per fold)
- [ ] Report mean ± std Macro F1 across folds
- [ ] Keep a **held-out 20-review lockbox** (or single stratified 70/30 split) if strict holdout is required for slides

**Files to change:** `scripts/train_models.py`, `scripts/evaluation.py`, optionally new `scripts/train_human_gold_cv.py`

**Expected lift:** Theme Macro F1 from ~30–44% toward **50–60%+**

---

### 1.2 Add star rating as a feature for sentiment

**Problem:** Text-only TF-IDF ignores strong star signal.

**Action:**
- [ ] Concatenate normalized `reviews.rating` (1–5) with TF-IDF features via `ColumnTransformer` or `scipy.sparse.hstack`
- [ ] Retrain sentiment models; evaluate Track A and Track B

**Files to change:** `scripts/preprocess_trainval.py`, `scripts/train_models.py`

**Expected lift:** Sentiment Macro F1 **+5–10 pts** on human gold

---

### 1.3 Richer text features

**Problem:** Char TF-IDF (2–4), 5,000 features, body text only.

**Action:**
- [ ] Combine **word + char** TF-IDF (`FeatureUnion`)
- [ ] Concatenate **title + body** before vectorizing
- [ ] Try `max_features=10000`, `min_df=2`, `sublinear_tf=True`

**Files to change:** `scripts/preprocess_trainval.py`, `scripts/text_utils.py`

**Expected lift:** **+2–5 pts** Macro F1 (both tasks)

---

### 1.4 Threshold tuning for Macro F1

**Problem:** Default 0.5 decision threshold favors majority class.

**Action:**
- [ ] On validation set (216 reviews), tune per-class thresholds to maximize Macro F1
- [ ] Apply tuned thresholds on human gold evaluation
- [ ] Log thresholds in `outputs/metrics/threshold_config.json`

**Files to change:** `scripts/evaluation.py`, new helper in `scripts/baselines.py`

**Expected lift:** Helps rare theme classes; modest overall gain

---

### Phase 1 checklist

- [ ] Implement 1.1–1.4
- [ ] Run `python scripts/run_post_labeling_pipeline.py`
- [ ] Refresh `outputs/metrics/` and `outputs/figures/`
- [ ] Verify `streamlit run dashboard.py` After tab reflects new numbers
- [ ] Add before/after comparison row to `Report.md` §5.3

---

## Phase 2 — Model & label quality (3–5 days)

### 2.1 Stronger classifiers + light hyperparameter search

**Action:**
- [ ] Add `LinearSVC` + `CalibratedClassifierCV`
- [ ] Add `SGDClassifier` (log loss, class_weight balanced)
- [ ] Optional: `XGBoost` / `LightGBM` on TruncatedSVD-reduced TF-IDF
- [ ] GridSearch on **validation only** (216 reviews): `C`, `n_estimators`, `max_depth`
- [ ] Never tune on human gold test set

**Expected lift:** **+3–8 pts** Macro F1 depending on task

---

### 2.2 Collapse or hierarchy for themes

**Problem:** 6-way classification with tiny classes (e.g. Customer service n=3 in gold).

**Action (choose one):**

| Option | Description |
|--------|-------------|
| **A — Merge rare classes** | e.g. Customer service + Delivery → `Service/Shipping` |
| **B — Two-stage hierarchy** | (1) Other vs Specific complaint → (2) fine-grained on complaints only |

- [ ] Document class mapping in `labeling/README.md`
- [ ] Re-evaluate with merged labels; compare Macro F1 fairly (old vs new mapping)

**Expected lift:** Often **+10–15 pts** Macro F1 on themes with clearer presentation story

---

### 2.3 Expand human labels (100 → 150–200)

**Action:**
- [ ] Sample 50–100 additional reviews stratified toward **Delivery**, **Customer service**, **Price complaint**
- [ ] Two annotators label; adjudicate disagreements
- [ ] Run `validate_human_labels.py` → `merge_human_labels.py`
- [ ] Optionally expand gold holdout or use CV only

> **See Phase 4** for scored/active-learning selection instead of manual random sampling.

**Expected lift:** More stable Macro F1 estimates; better rare-class recall

---

### 2.4 Tighten labeling guidelines

**Problem:** ~56% human–LLM theme agreement suggests ambiguous categories.

**Action:**
- [ ] Add 2–3 concrete examples per theme to `labeling/README.md`
- [ ] Adjudication session on 20 disagreement cases; update `LABELING_GUIDE.md` rules
- [ ] Re-label disputed gold rows if needed

**Expected lift:** Raises **ceiling** for all models; better science for the report

---

### Phase 2 checklist

- [ ] Implement chosen items from 2.1–2.4
- [ ] Update evaluation tracks if class schema changes
- [ ] Regenerate confusion matrices (`scripts/evaluation.py`)
- [ ] Update dashboard static “before” baselines if reporting methodology changes

---

## Phase 3 — LLM & hybrid routing (stretch)

### 3.1 Improve LLM theme prompts

**Action:**
- [ ] Refine Gemini prompt with definitions + few-shot examples
- [ ] Optional chain-of-thought: “Quote the phrase that indicates the theme”
- [ ] Re-run labeling on train split only; measure human agreement delta

**Files:** `llm_labeling.py`

---

### 3.2 Hybrid routing demo

**Architecture:**

```
Review → TF-IDF model (fast, local)
         ↓ confidence < 0.55
         → LLM fallback (Gemini)
         ↓ still ambiguous
         → human review queue
```

**Action:**
- [ ] Implement router in `scripts/predict.py` or extend `dashboard.py` Live Prediction tab
- [ ] Report accuracy/cost/latency tradeoff in report

**Value:** Strong presentation narrative even if pure ML Macro F1 stays moderate

---

### 3.3 Fine-tune small transformer (optional)

**Action:** DistilBERT fine-tuned on human labels (requires GPU/time)

**Recommendation:** Skip unless Phase 1–2 complete early and GPU available

---

## Phase 4 — TF-IDF feature engineering & active learning (review)

**Purpose:** Standalone phase for review. Documents whether to invest further in TF-IDF (you already use it) and how to **score unlabeled reviews** to select the most valuable samples for the next labeling/training round.

**Prerequisite:** Complete Phase 1.1 (human-label training target) first. TF-IDF and active learning improve efficiency but do not fix train/eval label mismatch on their own.

**Current baseline (`scripts/preprocess_trainval.py`):**

| Setting | Current value |
|---------|---------------|
| Vectorizer | `TfidfVectorizer` |
| Analyzer | Character n-grams |
| ngram_range | (2, 4) |
| max_features | 5,000 |
| Fit scope | Train split only (861 rows) |
| Text input | Body only (`reviews.text`) |

---

### 4.1 TF-IDF strategy — keep, enhance, or replace?

**Decision: Keep TF-IDF for the classical ML stack (NB, LR, RF). Do not drop it.**

| Option | Recommendation | Rationale |
|--------|----------------|-----------|
| **Keep char TF-IDF** | ✅ Yes | Robust to typos; already yields ~69% Macro F1 sentiment on validation |
| **Enhance TF-IDF** (§4.2) | ✅ **Primary action** | Low cost; word+char and title+body add signal without new infrastructure |
| **Replace with raw counts** | ❌ No | TF-IDF down-weights common terms; counts hurt on short reviews |
| **Replace with BERT/embeddings** | ⚠️ Stretch only | Needs GPU/time; unlikely to justify effort on 1,177 reviews for this course |

**What TF-IDF alone will not fix:**

1. Theme models trained on LLM labels but evaluated on human labels (→ Phase 1.1)
2. ~56% human–LLM theme agreement (label noise ceiling)
3. Rare classes (Customer service n=3 in gold)
4. Star rating ignored for sentiment (→ Phase 1.2)

**Expected lift from TF-IDF enhancements only:** **+2–5 pts** Macro F1. Pair with Phase 1.1–1.2 for meaningful gains.

---

### 4.2 TF-IDF enhancements (implementation spec)

Extends Phase 1.3 with explicit design choices and ablation notes for the report.

**Action:**
- [ ] **Title + body:** concatenate `reviews.title` + `reviews.text` before cleaning
- [ ] **Word + char union:** `FeatureUnion` with:
  - Word TF-IDF: `analyzer='word'`, `ngram_range=(1, 2)`, `max_features=8000`
  - Char TF-IDF: `analyzer='char'`, `ngram_range=(2, 4)`, `max_features=5000`
- [ ] **Vectorizer tuning:** `min_df=2`, `sublinear_tf=True`, `strip_accents='unicode'`
- [ ] **Optional numeric features:** stack normalized `reviews.rating` (sentiment task)
- [ ] **Ablation table (report):** char-only vs word+char vs word+char+title vs +rating

**Files to change:** `scripts/preprocess_trainval.py`, `scripts/text_utils.py`

**Do not:**
- Fit vectorizer on validation or gold rows
- Tune `max_features` on human gold (use validation only)

**Expected lift:** **+2–5 pts** Macro F1 (both tasks); sentiment may gain more with rating stacked

---

### 4.3 Active learning — score reviews for re-labeling

**Problem:** Manual expansion (Phase 2.3) picks reviews ad hoc. **Active learning** ranks unlabeled pool by informativeness so each labeling hour improves the model most.

**Candidate pool (eligible for selection):**

| Pool | Rows | Use |
|------|------|-----|
| Train split | 861 | Can add labels and retrain |
| Validation split | 216 | Can label for training **only if** you accept re-splitting; prefer keeping val fixed |
| Human gold holdout | 100 | ❌ **Never** add to training |

**Default pool:** train split (861) minus any rows already human-labeled.

---

### 4.4 Scoring strategies (combine 2–3)

Rank each unlabeled review with one or more scores; take the **top N** per batch.

| Strategy | Formula / rule | Best for |
|----------|----------------|----------|
| **Uncertainty** | `score = 1 - max(predict_proba)` | General — model is unsure |
| **Margin** | `score = 1 - (p_top1 - p_top2)` | Multi-class themes (close calls) |
| **Disagreement** | +1 if `model_pred ≠ llm_category`; +1 if `llm_sentiment ≠ sentiment` | Your dataset — already noisy |
| **Rare-class quota** | After ranking, enforce min picks per underrepresented theme | Delivery, Customer service |
| **Diversity (optional)** | Cluster TF-IDF; pick top uncertain per cluster | Avoid 50 near-duplicate “Other” reviews |

**Recommended combined score (theme batch):**

```
final_score = 0.5 * uncertainty + 0.3 * disagreement + 0.2 * rare_class_boost
```

**Recommended batch size:** 25–50 reviews per round; 2 annotators; adjudicate disagreements.

---

### 4.5 Active learning workflow

```mermaid
flowchart LR
    A[Pool: 861 train unlabeled] --> B[Train current best model]
    B --> C[Score: uncertainty + disagreement]
    C --> D[Apply rare-class quotas]
    D --> E[Export top 50 to labeling CSV]
    E --> F[Human label batch]
    F --> G[validate → merge → retrain]
    G --> H[Measure Track A Macro F1]
    H --> B
```

**Action:**
- [ ] Create `scripts/select_labeling_candidates.py`:
  - Input: trained model, vectorizer, manifest, `top_n`, task (`sentiment` | `theme`)
  - Output: `labeling/active_learning_batch_{date}.csv` with columns: `review_id`, `uncertainty`, `margin`, `disagreement`, `model_pred`, `llm_category`, `snippet`
- [ ] Never include `gold` split rows in export (assert in script)
- [ ] Document selection criteria in exported CSV README row or sidecar `.json`
- [ ] After labeling: merge via existing `validate_human_labels.py` → `merge_human_labels.py`
- [ ] Re-run pipeline; log Macro F1 delta in progress log

**Iteration rule:** Stop adding labels when Track A Macro F1 improvement &lt; 2 pts between batches, or labeling budget exhausted.

---

### 4.6 Evaluation & report language

**Metrics to track per active learning round:**

| Round | Labels added | Train size | Sentiment Macro F1 (gold) | Theme Macro F1 (gold) |
|-------|-------------|------------|---------------------------|------------------------|
| 0 (baseline) | 100 | 861 | 59.0% | 43.7% |
| 1 | +50 | 911 | TBD | TBD |
| 2 | +50 | 961 | TBD | TBD |

**Suggested report paragraph:**

> We applied **uncertainty-based active learning** to select the next labeling batch. Reviews were ranked by low model confidence and LLM–model disagreement, with stratified quotas for rare complaint themes. Each batch was human-labeled and merged into training without touching the 100-review gold holdout. This targets labeling effort toward ambiguous and underrepresented cases rather than random sampling.

---

### Phase 4 checklist

- [ ] Review §4.1 decision (keep TF-IDF) — confirm with team
- [ ] Implement §4.2 TF-IDF enhancements (or defer if Phase 1.3 already done)
- [ ] Implement `scripts/select_labeling_candidates.py`
- [ ] Run one active learning batch (50 reviews) with rare-class quotas
- [ ] Retrain and compare Track A before/after
- [ ] Add ablation + active learning tables to `Report.md` §5.5
- [ ] Optional: add “Suggested for labeling” panel in `dashboard.py`

---

## Recommended priority order

If time is limited, execute in this order:

1. **1.1** — Train theme on human labels + CV  
2. **1.2** — Star rating feature for sentiment  
3. **2.2** — Merge rare theme classes or two-stage hierarchy  
4. **4.3–4.5** — Active learning batch (replaces ad hoc Phase 2.3 sampling)  
5. **4.2** — TF-IDF word+char + title enhancements  
6. **3.2** — Hybrid LLM fallback demo  

---

## Commands reference

```bash
# Full pipeline after code changes
python scripts/run_post_labeling_pipeline.py

# Individual steps
python scripts/create_splits.py          # if splits change
python scripts/preprocess_trainval.py
python scripts/train_models.py
python scripts/evaluation.py
python scripts/select_labeling_candidates.py  # Phase 4 — after models trained

# Dashboard
streamlit run dashboard.py
```

---

## Deliverables when plan is complete

| Deliverable | Location |
|-------------|----------|
| Updated metrics (Track A/B/C) | `outputs/metrics/` |
| Updated figures | `outputs/figures/` |
| Retrained models | `*.pkl` (root) |
| Dashboard reflects new results | `dashboard.py` |
| Report section: improvement results | `Report.md` §5.4 (new) |
| Report section: TF-IDF ablation + active learning | `Report.md` §5.5 (Phase 4) |
| Active learning candidate export | `labeling/active_learning_batch_*.csv` |
| Slides talking points | `slides_guide.md` |

---

## Report narrative (use verbatim if helpful)

> Accuracy and Macro F1 dropped when we switched from LLM-as-ground-truth to human gold evaluation. This reflects **label noise**, **class imbalance**, and **train/eval target mismatch** for themes — not necessarily model regression. Improvements target label alignment, class consolidation, richer features, and hybrid LLM routing rather than inflated accuracy on circular benchmarks.

---

## Progress log

| Date | Phase | Done | Notes |
|------|-------|------|-------|
| 2026-08-05 | — | Plan written | Baseline: Sentiment Macro F1 59% (LR), Theme Macro F1 44% (LLM) on human gold |
| 2026-08-05 | 1.1–1.4, 4.5 | ☑ | Phase 1 implemented: word+char TF-IDF, rating feature, threshold tuning, CV, active learning export |
| | 1.1 | ☑ | 5-fold CV script; sentiment CV Macro F1 ~65%, theme ~21–26% |
| | 1.2 | ☑ | Star rating stacked with text for sentiment |
| | 1.3 | ☑ | Word+char TF-IDF, title+body, 13k features |
| | 1.4 | ☑ | LR threshold tuning on validation |
| | 4.5 | ☑ | `select_labeling_candidates.py` + batch CSVs |
| | 2.1 | ☐ | |
| | 2.2 | ☐ | |
| | 2.3 | ☐ | |
| | 2.4 | ☐ | |
| | 3.1 | ☐ | |
| | 3.2 | ☐ | |
| | 3.3 | ☐ | |
| | 4.1 | ☐ | TF-IDF strategy review |
| | 4.2 | ☐ | TF-IDF enhancements |
| | 4.3 | ☐ | Scoring strategies defined |
| | 4.4 | ☐ | Combined scoring formula |
| | 4.5 | ☐ | `select_labeling_candidates.py` + one batch |
| | 4.6 | ☐ | Report tables for AL rounds |

---

## Owners

| Phase | Suggested owner | Reviewer |
|-------|-----------------|----------|
| Phase 1 | TBD | TBD |
| Phase 2 | TBD | TBD |
| Phase 3 | TBD | TBD |
| Phase 4 | TBD | TBD |

*Update owners and progress log as work proceeds.*
