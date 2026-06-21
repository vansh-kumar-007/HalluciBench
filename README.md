[README.md](https://github.com/user-attachments/files/29175349/README.md)

# 🧠 HalluciBench: LLM Hallucination Benchmarking, Clustering, and Predictive ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Groq API](https://img.shields.io/badge/Orchestration-Groq%20%7C%20LiteLLM-green.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end data engineering and machine learning framework designed to evaluate, categorize, cluster, and predict structural hallucination footprint patterns across diverse Large Language Model (LLM) architectures.

Instead of generic broad knowledge probing, **HalluciBench** isolates model vulnerabilities across 6 specific linguistic, context-parsing, and factual failure domains, using an automated LLM-as-a-Judge pipeline to benchmark model reliability.

---

## 🚀 Key Empirical Insights
* **The Scale:** 95 evaluation rows across 5 distinct model engine families.
* **Overall Baseline:** An average benchmark hallucination rate of **28.4%**.
* **Most Vulnerable Category:** **Entity Confusion** proved to be the toughest failure domain, triggering a massive **60.0% hallucination rate** due to models conflating overlapping tokens (e.g., historical figures vs. corporate entities).
* **Model Vulnerability Index:** * `qwen-2.5-7b` demonstrated the highest failure frequency (**36.8% 🔴**) 
  * `llama-3.3-70b` established the strongest reliability index (**21.1% 🟢**), underscoring that raw scale helps mitigate hallucinations.

---

## 🗺️ System Pipeline Architecture

The implementation is broken down into a structured four-tiered processing sequence:

```text
┌─────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐     ┌──────────────────────┐
│  Data Inputs    │ ──> │  LLM-as-a-Judge      │ ──> │  Unsupervised NLP     │ ──> │ Supervised ML        │
│  (95 Rows / 5   │     │  (Llama-3.3-70b on   │     │  (TF-IDF + K-Means)   │     │ (Gradient Boosting   │
│   Model Types)  │     │   Groq Cloud Engine) │     │  Isolate Failure Type │     │  Stratified 5-Fold)  │
└─────────────────┘     └──────────────────────┘     └───────────────────────┘     └──────────────────────┘

```

### 1. The Evaluation Grid (Target Models)

The source matrix samples generations across:

* `llama-3.1-8b` (Meta AI)
* `llama-3.3-70b` (Meta AI)
* `gemini-2.5-flash-lite` (Google AI)
* `qwen-2.5-7b` (Alibaba Cloud)
* `cohere-command-a` (Cohere AI)

### 2. Failure Domain Schema

Prompts are engineered to expose weaknesses in six target categories:

* `factual_error`: Direct, confident historical/scientific misstatements.
* `fabricated_citation`: Generation of fictional URLs, DOI numbers, or ghost academic authors.
* `false_premise`: Validating if models detect and correct a false setup (e.g., *"Why did Edison invent the internet?"*).
* `entity_confusion`: Conflating distinct entities sharing identical nomenclature boundaries.
* `temporal_error`: Anachronistic sequencing and date mix-ups.
* `self_contradiction`: Inner-context consistency breaks where statements in an introduction clash with the summary.

---

## 🛠️ Repository Project Blueprint

The main runtime execution is split across these key processing sections:

### 📡 Automated LLM Grader (LLM-as-a-Judge)

Bypassing slow rate-limiting bottlenecks, the pipeline uses **Groq's hardware acceleration** to route prompts, expected answers, and model output texts to `llama-3.3-70b-versatile` under a strict deterministic signature. The system regex layer splits the raw markdown return buffer into clean data vectors:

* `is_hallucination`: Binary target variable (`0` = aligned, `1` = hallucinated).
* `severity`: Categorical impact ranking (`0` to `3`).
* `reason`: String explanation mapping the semantic mismatch.

### 🔬 Unsupervised Semantic Clustering

Focusing exclusively on the rows marked as active hallucinations, the pipeline builds a **TF-IDF Vectorizer Matrix** mapping 1-to-2 character n-gram frequencies. Running an unsupervised **K-Means Clustering** routine combined with a **Principal Component Analysis (PCA)** matrix reduction step uncovers four natural semantic groupings:

* **Cluster 0:** Temporal date anomalies and executive corporate timeline tracking errors.
* **Cluster 1:** Fabricated scientific study citations (e.g., medical claims, memory studies).
* **Cluster 2:** Literary entity confusion (e.g., misattributing poetry/authors).
* **Cluster 3:** Historical/Scientific figure confusion (e.g., Tesla vs. Napoleon).

### 🤖 Supervised Prediction Engine

The script constructs an analytical dataset by extracting linguistic and token features from the raw inputs:

* `response_length`: Total string character count.
* `word_count`: Discrete whitespace token separation count.
* `hedging_count`: Density frequency metrics tracking protective model vocabulary terms (*"likely"*, *"possibly"*, *"generally"*).
* Categorical Label-Encodings for `model` and `category` parameters.

#### Machine Learning Baseline Comparison

To avoid data leakage, algorithms are benchmarked using a **Stratified 5-Fold Cross-Validation Loop**:

* 🌲 **Random Forest Classifier**
* ⚡ **Gradient Boosting Classifier** *(Top Performer - CV F1 Score: 0.478)*
* 📉 **Logistic Regression**

---

## 📊 Dataset Dictionary & Features

The generated output data file `hallucination_dataset_labeled.csv` matches this scheme:

| Column Header | Format | Operational Definition |
| --- | --- | --- |
| `prompt_id` | `int64` | Relational integer identifier matching prompts across variants. |
| `category` | `object` | The engineered failure category class. |
| `prompt` | `object` | Input instruction sent to the baseline engine. |
| `model` | `object` | The precise model engine being evaluated. |
| `response` | `object` | Raw string output collected at `temperature=0.1`. |
| `expected_answer` | `object` | Gold-standard factually verified truth signature. |
| `notes` | `object` | Deep data science contextual notes detailing common failure modes. |
| `is_hallucination` | `int64` | Automated Judge Output Flag (`0` = True, `1` = Hallucination). |
| `severity` | `int64` | Evaluated impact classification tier (`0` to `3`). |
| `reason` | `object` | Dense string narrative detailing the logic behind the judge's score. |

---

## 📦 Local Installation & Setup

1. **Clone the Project Core:**
```bash
git clone [https://github.com/YOUR_USERNAME/hallucibench.git](https://github.com/YOUR_USERNAME/hallucibench.git)
cd hallucibench

```


2. **Establish Environment Dependencies:**
```bash
pip install -r requirements.txt

```


*(Note: Ensure `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `groq`, and `litellm` are installed).*
3. **Configure API Authorization Keys:**
Set your Groq Cloud credentials inside your local shell profile or Colab environment secrets:
```python
import os
os.environ["GROQ_API_KEY"] = "your_secret_groq_api_token_here"

```


4. **Run End-to-End Analytics Execution:**
Execute your Jupyter Environment notebook `ai-model-hallucination-benchmark-eda-clustering-ml.ipynb` to re-generate the full modeling pipeline, save your plots, and output the finalized labeled tabular data asset.

---

## 🔮 Future Development Roadmap

1. **Sample Scaling:** Expand from 95 testing pairs to a production-grade 500+ testing pair configuration matrix.
2. **Advanced NLP Embeddings:** Replace traditional TF-IDF extraction arrays with context-aware semantic sentence embeddings (`all-MiniLM-L6-v2`) to capture deeper semantic relationships.
3. **Few-Shot Prompt Mitigations:** Introduce counter-measure prompt structures (few-shot prompting, chain-of-thought) to observe how effectively they reduce hallucination frequencies across different models.

---

*Developed by Vansh Kumar | Optimized for Kaggle Benchmarking & Open-Source LLM Reliability Research.*

```

***
