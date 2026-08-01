# PowerPoint Presentation Guide
## Sentiment and Theme Classification of Product Reviews

This document provides a slide-by-slide content draft, visual guidelines, and speaker notes. Copy this text directly into your PowerPoint slides and divide the presentation sections between the two team members.

---

## Slide 1: Project Title and Team Members

### Slide Content
* **Title**: Multi-Task Automated Classification of E-Commerce Product Feedback
* **Team Members**: [Insert Member 1 Name] and [Insert Member 2 Name]
* **Course**: Data Mining (Graduate Level)
* **Project Overview**: An end-to-end data mining pipeline using Vertex AI Gemini to generate training labels from raw text and training classical classifiers (Naive Bayes, Logistic Regression, Random Forest) to categorize product reviews by sentiment and operational complaint themes.

### Speaker Notes
* **Speaker**: Both members introduce themselves.
* **Talking Points**: "Welcome to our final project presentation. Today we will walk you through our system for automated classification of product reviews. We integrated Large Language Models for data labeling and compared them with fast, local classical machine learning models to see how companies can deploy high-speed, zero-cost triage systems in production."

---

## Slide 2: Problem Definition

### Slide Content
* **The Business Challenge**: E-commerce sites receive millions of reviews. Reading them manually to flag issues is slow, expensive, and does not scale.
* **The "Signal vs. Noise" Problem**: Customer reviews often mix product feedback ("the battery died") with operational complaints ("shipping was delayed"). Product managers struggle to separate this signal from the operational noise.
* **Who Benefits**: 
  * Support teams (instant routing of operational complaints).
  * Product developers (instant identification of product defects).
  * Marketing teams (sentiment tracking).
* **The Classification Tasks**:
  1. **Sentiment Task**: Predict Positive, Neutral, or Negative (3-class classification).
  2. **Theme/Complaint Task**: Predict Product quality issue, Feature request, Price complaint, Customer service issue, or Other (5-class classification).

### Speaker Notes
* **Speaker**: Member 1
* **Talking Points**: "The main problem we are trying to solve is the inefficiency of manual review triage. A customer support agent shouldn't have to read through thousands of positive reviews just to find the five customers complaining about shipping delays or defective hardware. Our system classifies reviews in two dimensions: core sentiment (positive, neutral, negative) and operational theme. This helps automatically route reviews to the right departments."

---

## Slide 3: Dataset Selection & Description

### Slide Content
* **Source**: Datafiniti Amazon Consumer Reviews Dataset.
* **Size**: 1,177 clean customer reviews of Amazon Kindle, Echo, and Fire TV devices.
* **Record Representation**: Each row represents an individual customer review.
* **Core Columns Used**:
  * `reviews.text` (Unstructured text body of the review)
  * `reviews.title` (Review headline)
  * `reviews.rating` (Numerical star rating, 1 to 5)
  * `sentiment` (Ground truth sentiment derived from rating: 1-2 = Negative, 3 = Neutral, 4-5 = Positive)
* **Dataset Characteristics**: Real-world e-commerce data with mixed user tones and language variations.
* **Limitations**: Highly imbalanced (84% positive, 11% negative, 5% neutral).

### Speaker Notes
* **Speaker**: Member 1
* **Talking Points**: "We utilized the Datafiniti Amazon Consumer Reviews dataset. We selected a clean subset of 1,177 reviews of Amazon Devices. Each record contains the review text, title, and rating. The major challenge of this dataset is class imbalance: like most e-commerce data, customers are generally happy, so positive reviews heavily dominate. We will show how this imbalance affected our classical classifiers."

---

## Slide 4: Data Preprocessing and Feature Preparation

### Slide Content
* **Cleaning Steps**:
  1. Lowercased all review text.
  2. Stripped punctuation and special characters to normalize features.
  3. Preserved international characters to keep multilingual review integrity.
* **Representation Learning (TF-IDF)**:
  * Converted text into numerical vectors using a TF-IDF Vectorizer.
  * Configured for **Character N-Grams** (n-gram range: 2 to 4 characters, 5,000 maximum features).
* **Why Character N-Grams?**
  * Resilient against typos (e.g., "broken", "brokene", "brokn" share character substrings).
  * Robust against mixed languages without needing translation.
* **Train-Test Split**:
  * 80% Training Set (941 reviews) / 20% Testing Set (236 reviews).
  * **Stratified Split**: Preserved the class ratio of Positive/Neutral/Negative across both sets.

### Speaker Notes
* **Speaker**: Member 2
* **Talking Points**: "For preprocessing, we cleaned the text and applied a character-level TF-IDF vectorizer extracting character ranges of 2 to 4 letters. This is a critical design choice. Unlike word-level vectors, character n-grams are highly robust to user typos and spelling mistakes, which are common in reviews. We split the data into 80% train and 20% test using a stratified split to ensure the minor negative and neutral classes were represented in both sets."

---

## Slide 5: AI-Assisted Labeling & Validation

### Slide Content
* **AI Tool Used**: Google Cloud Vertex AI API (`gemini-2.5-flash`).
* **Task**: Zero-shot categorization of reviews into Sentiment and Theme.
* **Prompt Design**: Strictly structured system instructions requesting a JSON response with keys `llm_sentiment` and `llm_category`.
* **Validation Method**: Mapped numerical user star ratings to sentiment classes and compared them with LLM predictions.
* **Validation Metrics**:
  * **Sentiment Agreement Accuracy**: 82.24%
  * **Cohen's Kappa Score**: 0.4631 (Moderate Agreement)
* **Risk/Insight**: Users often select a 5-star rating out of habit but write a neutral text (e.g., "It works, nothing special"). The LLM correctly overrides these to "Neutral" based on text tone, making the moderate Kappa score mathematically defensible.

### Speaker Notes
* **Speaker**: Member 2
* **Talking Points**: "To label the complaint categories and verify sentiment, we used Vertex AI Gemini. We designed a JSON prompt instructing the LLM to analyze the review text. To validate the AI labels, we compared them to the user star ratings. We achieved 82.24% accuracy and a Cohen's Kappa score of 0.4631. This moderate agreement is actually a positive finding. It shows the LLM is reading textual nuance—for example, marking a review as Neutral based on the text even if the user selected a 5-star rating."

---

## Slide 6: Classification Models & Implementation

### Slide Content
* **Comparison Setup**: We compared three classical machine learning algorithms trained on the LLM-generated labels:
  1. **Multinomial Naive Bayes (NB)**: Probabilistic classifier based on term frequencies. Good text classification baseline.
  2. **Logistic Regression (LR)**: Linear model identifying decision boundaries. Works well with high-dimensional character n-grams.
  3. **Random Forest (RF)**: Ensemble of 100 Decision Trees. Non-linear, robust against feature correlation.
* **Zero-Shot LLM Comparison**:
  * We also compared the trained classical models against the zero-shot LLM predictions on the holdout test set to evaluate performance vs. deployment cost.

### Speaker Notes
* **Speaker**: Member 1
* **Talking Points**: "We trained three classical models on the LLM labels: Naive Bayes, Logistic Regression, and Random Forest. Our goal was to see if a small classical model, trained on labels generated by an AI, could perform close to the AI itself on new data, saving us cloud costs in production."

---

## Slide 7: Sentiment Classification Performance

### Slide Content
* **Evaluation on holdout test set (236 reviews)** against human user ratings:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes** | 83.47% | 27.82% | 33.33% | 30.33% | 75.96% |
| **Logistic Regression** | **85.59%** | **70.68%** | 48.69% | 51.16% | 81.52% |
| **Random Forest** | 83.90% | 56.08% | 59.66% | 55.10% | 83.03% |
| **LLM (Zero-Shot)** | 84.75% | 58.13% | **68.58%** | **59.83%** | **83.94%** |

* **Key Sentiment Findings**:
  * Zero-shot LLM has the highest Macro F1 (59.83%), showing it handles the minority "Neutral" and "Negative" classes best.
  * Random Forest is the best classical model (55.10% Macro F1).
  * Naive Bayes suffers from the class imbalance, predicting "Positive" for almost everything, resulting in 0% recall for Negatives and Neutrals.
  * **Visual Graphic**: Insert `sentiment_confusion.png` (Heatmap showing how Random Forest maps targets to predictions).

### Speaker Notes
* **Speaker**: Member 1
* **Talking Points**: "Here are the sentiment results. If you look at standard accuracy, all models look great—around 84% to 85%. But this is misleading due to class imbalance. If we look at Macro F1, which weighs all classes equally, Naive Bayes drops to 30.33% because it fails to detect negative and neutral reviews. Logistic Regression and Random Forest perform much better, with Random Forest achieving 55.10% Macro F1, very close to the zero-shot LLM's 59.83%."

---

## Slide 8: Theme/Complaint Classification Performance

### Slide Content
* **Evaluation on holdout test set (236 reviews)** against LLM target categories:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes** | 72.03% | 18.01% | 25.00% | 20.94% | 60.32% |
| **Logistic Regression** | 76.27% | 62.81% | 31.96% | 32.72% | 68.96% |
| **Random Forest** | **77.12%** | **91.28%** | **40.84%** | **46.81%** | **70.65%** |

* **Key Theme Findings**:
  * Random Forest is the strongest model for theme classification, with a Macro F1 of 46.81%.
  * The "Other" class dominates (170 reviews), making theme classification a highly imbalanced 5-class problem.
  * Classical models successfully learned theme associations (like detecting "battery" or "screen" as Product Quality issues) from character TF-IDF.
  * **Visual Graphic**: Insert `category_confusion.png` (Heatmap showing class-specific true vs. predicted counts).

### Speaker Notes
* **Speaker**: Member 2
* **Talking Points**: "For theme classification, we are predicting the 5 operational categories. Again, Random Forest was the best model, achieving 77.12% accuracy and 46.81% Macro F1. Because 'Other' (general reviews) represented 72% of the test set, Naive Bayes again struggled to detect the specific complaints, while Random Forest successfully leveraged decision trees to identify product quality issues and feature requests."

---

## Slide 9: Cost, Latency & Operational Discussion

### Slide Content
* **Operational Latency**:
  * Classical models: **Less than 1 millisecond** per review (Instant local execution).
  * Large Language Model: **~800 milliseconds** per review (Requires cloud roundtrip).
* **Financial Cost**:
  * Classical models: **$0.00** (Free to host and run locally).
  * LLM labeling: **~$0.13 USD** for the entire 1,177 dataset (Vertex AI API rates).
* **Recommended Production Architecture**:
  * Use the **LLM as an offline labeler** to compile training datasets.
  * Train a **Random Forest Classifier** on the LLM-generated labels.
  * Deploy the **Random Forest locally** to classify live production reviews. This preserves 92% of the LLM's sentiment quality with zero operational cost and sub-millisecond speeds.

### Speaker Notes
* **Speaker**: Member 2
* **Talking Points**: "In a real business, we have to consider speed and cost. LLMs are slow and require paid cloud calls. Classical models run in under a millisecond and cost nothing. Our recommendation is a hybrid approach: use the LLM to label your historical reviews, and use that labeled data to train a Random Forest model. This gives you high-quality labeling with zero runtime cost."

---

## Slide 10: Conclusion & Reflection

### Slide Content
* **Main Result**: Successfully built a hybrid AI-assisted text classification pipeline. Random Forest proved to be the most robust classical model, closely matching zero-shot LLM performance.
* **Key Lesson Learned**: Macro F1 is the only reliable metric for e-commerce sentiment analysis. Standard accuracy will hide a model's failure to detect customer complaints.
* **Limitations**: High class imbalance in the training data limits model recall for rare complaints (like pricing).
* **Future Work**: Implement synthetic data oversampling (SMOTE) to boost minority class training, and transition to deep learning representations (DistilBERT).

### Speaker Notes
* **Speaker**: Both members conclude.
* **Talking Points**: "In conclusion, our pipeline successfully automated review triage. The main lesson is that you must look at F1-scores, not accuracy, on imbalanced data. For future work, we plan to test oversampling techniques to make the classical models even better at catching rare complaints. Thank you, and we are ready for your questions."
