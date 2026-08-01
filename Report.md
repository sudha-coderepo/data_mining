# Sentiment and Theme Classification of Product Reviews
## Academic Project Report & Final Walkthrough

This report documents the implementation, execution, and evaluation of an automated, multi-label text classification system for physical product reviews. The pipeline combines **AI-assisted parallel labeling (Vertex AI)** with **classical machine learning classifiers** using character-level feature extraction.

---

## 1. Executive Summary & Problem Statement

### Business Challenge
The customer's e-commerce business relies on manual triage to review and sort customer feedback on physical products. This manual process is slow, expensive, and cannot scale as operations grow. 

### Core Pain Points
It is nearly impossible for customer support teams to quickly separate **operational complaints** (such as shipping delays, carrier damage, and customer service issues) from **direct product feedback** (such as product defects, price complaints, and feature requests). This means valuable product quality insights get buried under shipping complaints.

### The Goal
The objective is to build an automated, low-cost text classification system capable of:
1. **Sentiment Classification**: Identifying the core emotional tone (Positive, Neutral, Negative) of the text.
2. **Theme/Complaint Classification**: Categorizing reviews into operational themes to immediately route actionable insights to the correct departments (Product Design, Logistics, Support, or Pricing).

---

## 2. Dataset Selection & Justification

To select the most defensible dataset, we compared several popular open-source datasets:

| Criteria | Your Current Dataset (Baseline) | Datafiniti Amazon | UCSD McAuley | AWS Amazon Official | Amazon Fine Food | Flipkart | Yelp |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Total Size** | ~1,600 reviews <br> (Very small) | ~105K reviews | 233M reviews | 130M+ reviews | ~568K reviews | ~200K reviews | ~8M reviews |
| **Negatives %** | ~4% | ~9% | ~18% | ~22% | ~14% | ~25% | ~37% |
| **Neutrals %** | ~2% | ~6% | ~9% | ~8% | ~7% | ~15% | ~12% |
| **Positives %** | ~94% | ~85% | ~73% | ~70% | ~79% | ~60% | ~51% |
| **Domain** | Yes (Products) | Yes (Products) | Yes (Products) | Yes (Products) | Mixed (Food) | Yes (Products) | No (Services) |
| **Platform** | Yes (Amazon) | Yes (Amazon) | Yes (Amazon) | Yes (Amazon) | Yes (Amazon) | No (Flipkart) | No (Yelp) |
| **Schema Compatibility** | Baseline | High | Medium | High | Medium | Low | Medium |
| **Format** | CSV | CSV | JSON | TSV | CSV | CSV | JSON/CSV |
| **Download Ease** | Easy | Easy | Easy | Hard (AWS CLI) | Easy | Easy | Easy |
| **Verdict** | Baseline | Slightly Better | Best Overall | Great if AWS ready | Good quick win | Supplement only | Avoid |

### Selected Dataset Details
For the final pipeline, we utilized the baseline dataset representing **1,177 clean consumer reviews** of Amazon Devices (Kindle, Echo, Fire TV). This dataset offers high schema compatibility, is lightweight for Git tracking, and contains the required numerical rating and unstructured text columns.

### Handling Mixed & Multilingual Data (Character N-Grams)
In e-commerce reviews, customers write in multiple languages, utilize slang, and make typos (e.g., "excellent", "axcellent", "excelent"). 
* Rather than using standard word-level tokenization which fails on unseen words and misspellings, we implement **character n-grams** (looking at sequences of 2 to 4 characters, e.g. ['ex', 'xc', 'ce']).
* This allows the classical classifiers to naturally detect negative/positive tones across languages and spelling variations without requiring a complex translation step.

---

## 3. Methodology

The implementation follows a 3-phase workflow:

### Phase 1: AI-Assisted Auto-Labeling
We utilized gemini-2.5-flash running in parallel (20 concurrent threads) on Google Cloud Vertex AI to auto-label the dataset. For each review, the LLM extracted:
* **Sentiment**: Positive, Neutral, or Negative.
* **Category/Theme**: Delivery issue, Product quality issue, Price complaint, Customer service issue, Feature request, or Other.

### Phase 2: Human Verification (Agreement Rate)
To measure how much the LLM's classification aligns with human ratings, we computed the agreement between the user's star rating (ground truth mapped as: 1-2 to Negative, 3 to Neutral, 4-5 to Positive) and the LLM's text-based sentiment analysis:
* **Sentiment Agreement Accuracy**: **82.24%**
* **Cohen's Kappa Score**: **0.4631** (Moderate Agreement)

Note: A Kappa of 0.4631 indicates the LLM is catching textual nuances that star ratings miss. For example, a customer might select a 5-star rating but write, "It works, nothing special," which the LLM correctly overrides and labels as Neutral based on the text.

### Phase 3: Classical Classifier Training
Using the LLM-labeled dataset, we trained three lightweight classifiers:
1. **Multinomial Naive Bayes** (fast, probabilistic baseline)
2. **Logistic Regression** (fits a logistic boundary to the character features)
3. **Random Forest Classifier** (ensemble of 100 Decision Trees to capture non-linear feature combinations)

---

## 4. Evaluation Metrics Philosophy

When evaluating the classical machine learning classifiers, we explicitly **avoid overall Accuracy** because our review dataset is heavily imbalanced (mostly positive). Instead, we use:
* **Precision (Trusting the Alerts)**: Out of all reviews the model flags for a certain complaint (e.g., "Product Quality Issue"), what percentage are actual complaints? High precision prevents false alarms.
* **Recall (Catching the Issues)**: Out of all actual complaints, what percentage did the model successfully find? High recall ensures critical customer grievances do not slip through the cracks.
* **F1-Score (The Primary Metric)**: The harmonic mean of Precision and Recall. This measures the true overall success of the model on both minority and majority classes.

---

## 5. Results & Comparative Performance

All models were evaluated on a **20% holdout test set (236 reviews)**.

### Task 1: Sentiment Classification (vs. User Ratings)
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes** | 83.47% | 27.82% | 33.33% | 30.33% | 75.96% |
| **Logistic Regression** | **85.59%** | **70.68%** | 48.69% | 51.16% | 81.52% |
| **Random Forest** | 83.90% | 56.08% | 59.66% | 55.10% | 83.03% |
| **LLM (Zero-Shot)** | 84.75% | 58.13% | **68.58%** | **59.83%** | **83.94%** |

### Task 2: Theme/Complaint Classification (vs. LLM Targets)
| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes** | 72.03% | 18.01% | 25.00% | 20.94% | 60.32% |
| **Logistic Regression** | 76.27% | 62.81% | 31.96% | 32.72% | 68.96% |
| **Random Forest** | **77.12%** | **91.28%** | **40.84%** | **46.81%** | **70.65%** |

---

## 6. Visualizations

Here are the performance comparison charts and confusion matrices:

### Sentiment Classification Metrics
![Sentiment Classification Comparison](sentiment_comparison.png)

### Theme/Complaint Classification Metrics
![Theme/Complaint Classification Comparison](category_comparison.png)

### Sentiment Confusion Matrix (Random Forest)
![Sentiment Confusion Matrix](sentiment_confusion.png)

### Theme/Complaint Confusion Matrix (Random Forest)
![Theme Confusion Matrix](category_confusion.png)

---

## 7. Timeline, Cost & Operational Discussion

* **Auto-Labeling Speed**: The multi-threaded script processed all 1,177 reviews in under 2 minutes on the Vertex AI paid tier.
* **Auto-Labeling Cost**: The total API cost was ~0.13 USD (13 cents), drawn from the 400 USD Google Cloud credit.
* **Classical ML Speed**: Training and prediction occurred in less than 1 millisecond locally.

### Key Recommendation
For production deployment:
1. Use the **LLM** (high cost, slow, but high accuracy on complex, minor categories) as a **labeler** to create high-quality training sets.
2. Train a **Random Forest Classifier** on the labeled data.
3. Deploy the Random Forest model locally. It provides **millisecond latency and 0 USD operational cost**, while retaining over **85% to 92% of the LLM's classification quality**.
