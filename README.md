# 🔍 FactLens: Detecting Fake News & Political Bias with Explainable ML

> **CAP 5610 — Introduction to Machine Learning | Group 9 | Florida International University | April 2026**  
> Mauricio Velasquez · Valentina Kloster · John Corbacho Soubal

---

## 🎥 Demo Video

[Watch the project presentation](https://drive.google.com/file/d/1Pi2VjHoTQWcFTQxkptiS0_Yk892yl39U/view?usp=sharing)

---

## Overview

FactLens is a dual-task machine learning system that detects **fake news** and **political bias** in news articles simultaneously, with full explainability. Given any article text, FactLens returns:

- ✅ **Fake or Real** — with confidence score and top contributing words
- 🔵🔴 **Left or Right leaning** — with confidence score and top contributing words

Built with Logistic Regression (TF-IDF) and fine-tuned DistilBERT, with explainability via LR coefficients and Integrated Gradients.

---

## Key Results

| Model | Fake vs Real | Left vs Right |
|-------|-------------|---------------|
| LR — Original (TF-IDF) | 98.52% | — |
| LR — Debiased | 97.50% | 87.1% |
| LR — Enhanced (+5 features) | 98.33% | — |
| DistilBERT — Fine-tuned | **99.97%** | 86.6% |

**Key finding:** Fake news and political bias are linguistically orthogonal (Pearson r ≈ 0, zero shared top-20 vocabulary) — they require separate, independent classifiers.

---

## Project Structure

```
FactLens/
│
├── fake_news_detection/           # Dataset 1 notebooks (Fake vs Real)
│   ├── Step2_EDA.ipynb
│   ├── Step3_Cleaning.ipynb
│   ├── Step3b_Features.ipynb
│   ├── Step4_TFIDF.ipynb
│   ├── Step5_Split.ipynb
│   ├── Step6_LogisticRegression.ipynb
│   ├── Step6b_LR_Enhanced.ipynb
│   ├── Step7_Evaluation.ipynb
│   ├── Step8_Explainability_LR.ipynb
│   ├── Step8b_SourceBias_Experiment.ipynb
│   ├── Step9_DistilBERT.ipynb
│   ├── Step10_LIME_SHAP.ipynb
│   ├── Step10b_Explainability_DistilBERT.ipynb
│   ├── Step11_LR_vs_DistilBERT.ipynb
│   └── Step12_CrossTask_Analysis.ipynb
│
├── political_bias/                # Dataset 2 notebooks (Left vs Right)
│   ├── Step1_LeftVsRight_LogisticRegression.ipynb
│   └── Step2_LeftVsRight_DistilBERT.ipynb
│
├── demo/
│   └── FactLens_Demo.ipynb        # Live Gradio demo
│
├── presentation/
│   └── FactLens_Presentation.pptx
│
├── report/
│   └── FactLens_Report.pdf
│
├── data/
│   └── README.md                  # Download instructions for datasets
│
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Source | Articles | Task |
|---------|--------|----------|------|
| Fake vs Real News | [Kaggle — Bisaillon](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) | 44,898 → 38,590 after cleaning | Binary: Fake / Real |
| Political Bias | [Kaggle — cl0ud0](https://www.kaggle.com/datasets/cl0ud0/news-political-bias-classification-dataset) | 13,366 | Binary: Left / Right |

> **Note:** Datasets are not included in this repository due to size. Download them from the Kaggle links above and place them in `data/` respectively.

---

## Pipeline

```
Raw CSV
   ↓
Cleaning (lowercase → stopwords → lemmatization → deduplicate)
   ↓
Feature Extraction (sentiment, subjectivity, readability, punctuation)
   ↓
TF-IDF Vectorization (50,000 features)
   ↓                          ↓
Logistic Regression        DistilBERT Fine-tuning
   ↓                          ↓
LR Coefficients            Integrated Gradients
   ↓                          ↓
         Cross-Task Analysis
```

---

## Running the Demo

The demo runs in Google Colab. Open `demo/FactLens_Demo.ipynb` and run all cells.

```python
# Cell 1 — Mount Drive
from google.colab import drive
drive.mount("/content/drive")

# Cell 2 — Install dependencies
!pip install gradio vaderSentiment -q

# Cell 3 — Load models and launch
# (loads both LR models + TF-IDF vectorizers, launches Gradio with share=True)
```

A public Gradio link will be generated. Paste any article text and get instant results for both tasks.

---

## Notebooks Guide

All notebooks are designed to run in **Google Colab** with Google Drive mounted. Run them in order within each dataset. Update `DATA_PATH` in the first cell of each notebook to match your Google Drive folder.

**Dataset 1 (Fake vs Real) — `fake_news_detection/`**

| Step | Notebook | Description |
|------|----------|-------------|
| 2 | Step2_EDA | Exploratory data analysis |
| 3 | Step3_Cleaning | Text cleaning pipeline |
| 3b | Step3b_Features | Linguistic feature extraction |
| 4 | Step4_TFIDF | TF-IDF vectorization |
| 5 | Step5_Split | Train/test split |
| 6 | Step6_LogisticRegression | LR training |
| 6b | Step6b_LR_Enhanced | LR + features training |
| 7 | Step7_Evaluation | Model evaluation |
| 8 | Step8_Explainability_LR | LR coefficient analysis |
| 8b | Step8b_SourceBias_Experiment | Debiasing experiment |
| 9 | Step9_DistilBERT | DistilBERT fine-tuning (GPU required) |
| 10 | Step10_LIME_SHAP | LIME attempt + limitation documentation |
| 10b | Step10b_Explainability_DistilBERT | Integrated Gradients |
| 11 | Step11_LR_vs_DistilBERT | Model comparison |
| 12 | Step12_CrossTask_Analysis | Cross-task orthogonality analysis |

**Dataset 2 (Left vs Right) — `political_bias/`**

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | Step1_LeftVsRight_LogisticRegression | LR + grid search |
| 2 | Step2_LeftVsRight_DistilBERT | DistilBERT fine-tuning |

---

## Setup

```bash
# Clone the repo
git clone https://github.com/your-username/FactLens.git
cd FactLens

# Install dependencies
pip install -r requirements.txt
```

For DistilBERT training (Steps 9 and 2), use a GPU runtime (A100 recommended).

---

## Explainability

Three explainability approaches were used:

- **LR Coefficients** — Each word's weight directly quantifies its contribution. Positive → Fake/Right. Negative → Real/Left.
- **LIME** — Attempted but produced near-zero scores due to model confidence >99.9%. Documented as a known limitation of perturbation-based methods on highly confident models.
- **Integrated Gradients** — Gradient-based attribution on DistilBERT. Revealed that "reuters" alone carried an attribution score of 0.86 for the Real class — the single strongest source bias signal in the project.

---

## Key Findings

1. **Source bias is real and measurable.** "reuters" was the top Fake indicator in LR (coefficient +19.60) — the model learned Reuters wire formatting, not misinformation. Debiasing caused only a 1.02% accuracy drop, confirming genuine content learning underneath.

2. **LIME fails on confident models.** Perturbation-based explainability requires model uncertainty. When confidence exceeds 99.9%, no single word removal shifts the prediction enough to measure. Integrated Gradients is the correct alternative.

3. **Political bias is keyword-driven.** LR and DistilBERT tie at ~87% on the bias task. Deep contextual modeling provides no advantage — the signal is in the vocabulary, not the sentence structure.

4. **Fake news and political bias are orthogonal.** Pearson r = −0.049 (not significant), zero shared top-20 vocabulary across all class pairs. They are independent linguistic phenomena requiring independent models.

---

## Authors

Mauricio Velasquez · Valentina Kloster · John Corbacho Soubal  
CAP 5610 — Florida International University, April 2026

---

## License

This project was developed for academic purposes as part of CAP 5610 at Florida International University.