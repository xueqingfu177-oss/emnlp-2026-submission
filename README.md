# Claim-DER: Multi-Agent Framework for Patent Claim Generation

This repository contains the official anonymous implementation and datasets for our EMNLP submission: **"Beyond Mere Generation: Graph-Centered Multi-Agent Framework for Patent Claim Generation"**. 

Our proposed framework, **Claim-DER**, tackles two critical failures in large language models for patent drafting: missing deep-tail technical details and broken claim dependencies. It separates long-text detail gathering from structure planning via a collaborative Drafter-Examiner-Reviser pipeline.

## 📂 Repository Structure

The project is organized to ensure seamless reproducibility:

```text
├── data/
│   ├── chemistry.json       # Patent-CR dataset (English, Chemistry domain)
│   └── lithography.json     # Patent-LG-zh dataset (Chinese, Lithography domain)
├── src/
│   ├── eval/
│   │   ├── DAR.py           # Dependency-Aware Regression (DAR) evaluation
│   │   ├── LLM-as-a-judge.py# N-way peer review via Qwen-Max
│   │   ├── PAR.py           # Position-Aware Recall (PAR) evaluation
│   │   └── Rouge.py         # Standard lexical ROUGE metric calculation
│   └── models/
│       ├── Drafter.py             # Stage 1: Initial claim tree generation
│       └── Examiner-Reviser.py    # Stages 2 & 3: Detail extraction & conservative rewriting
├── README.md
└── requirements.txt
```

## 🛠️ Environment Setup

We recommend using **Python 3.9+** for optimal compatibility. To set up the environment, please run:

```bash
# 1. Install required python packages
pip install -r requirements.txt

# 2. Download language models for spaCy (Required for PAR and DAR metrics)
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

**API Key Configuration:**
Our framework utilizes the DashScope API for generating claims and evaluating via LLM-as-a-judge. Please set your API key as an environment variable before running the scripts:
```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

## 🚀 Reproducing the Results

The inference pipeline is strictly aligned with the three-stage architecture of Claim-DER.

### Step 1: Claim Tree Construction (Drafter)
The Drafter processes the technical disclosure to establish a foundational structural layout.
```bash
python src/models/Drafter.py
```
*Outputs:* `./data/output-zero-shot-chemistry.json`

### Step 2: Inspection & Completion (Examiner & Reviser)
The Examiner scans for missing deep-tail features, while the Reviser safely integrates these details without corrupting the initial reference chains.
```bash
python src/models/Examiner-Reviser.py
```
*Outputs:* `./data/output-two-agents-chemistry.json`

## 📊 Evaluation

We provide four comprehensive evaluation scripts corresponding to the metrics discussed in the paper. By default, they evaluate the English (`chemistry`) dataset. You can toggle the `LANGUAGE` flag inside the scripts to evaluate the Chinese (`lithography`) dataset.

**1. Position-Aware Recall (PAR)**
To measure the extraction of deep-tail features:
```bash
python src/eval/PAR.py
```

**2. Dependency-Aware Regression (DAR)**
To precisely measure structural compliance and reference tracking:
```bash
python src/eval/DAR.py
```

**3. ROUGE Recall**
To quantify lexical coverage:
```bash
python src/eval/Rouge.py
```

**4. LLM-as-a-Judge**
To conduct an N-way blind peer review using Qwen-Max across 5 expert dimensions (Feature Completeness, Conceptual Clarity, Terminology Consistency, Logical Linkage, Overall Quality):
```bash
python src/eval/LLM-as-a-judge.py
```
