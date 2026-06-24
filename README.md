# Smart Digital Marketing Multi-Agent System (MAS)

AI-powered Multi-Agent System for Digital Marketing Analytics, Campaign Optimization, and Advertising Decision Recommendation using Local LLMs with Ollama.

---

# Overview

This project adapts a Smart Manufacturing MAS architecture into a Digital Marketing and Advertising AI Agent System.

The system uses multiple AI agents to:

- load marketing datasets
- preprocess customer and campaign data
- select machine learning models dynamically
- analyze campaign performance
- predict conversion outcomes
- generate advertising decision recommendations
- support human-in-the-loop review

The orchestration is powered by a local LLM planner using Ollama.

---

# System Architecture

```text
LLM Planner Agent
        |
        v
Data Loader Agent
        |
        v
Preprocessing Agent
        |
        v
Dynamic Analysis Agent
        |
        v
Optimization / Recommendation Agent
        |
        v
Human-in-the-Loop Review
```

---

# Features

- Multi-agent workflow orchestration
- Local LLM support using Ollama
- Digital marketing campaign analysis
- Conversion prediction
- Campaign performance modeling
- CTR and engagement analysis
- Advertising recommendation generation
- Human-in-the-loop approval
- Intelligent reporting
- Publication-ready logs and summaries

---

# Project Structure

```text
smart_manufacturing_mas_code/
│
├── agents/
│   ├── data_loader_agent.py
│   ├── preprocessing_agent.py
│   ├── dynamic_analysis_agent.py
│   ├── optimization_agent.py
│   ├── llm_planner_agent.py
│   └── local_llm_agent.py
│
├── utils/
│   ├── tool_decider.py
│   ├── reporting.py
│   ├── intelligent_summarization.py
│   └── hitl_interface.py
│
├── data/
│   └── Digital Marketing Campaign Dataset/
│
├── logs/
│
├── main_llm.py
└── README.md
```

---

# Requirements

Recommended Python version:

```text
Python 3.10+
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Additional recommended packages:

```bash
pip install ollama transformers python-dotenv
```

---

# Ollama Setup

Download and install Ollama:

```text
https://ollama.com/download
```

Check installation:

```bash
ollama list
```

Pull recommended local model:

```bash
ollama pull qwen3:4b
```

Alternative model:

```bash
ollama pull mistral:7b
```

---

# Dataset

Place the CSV dataset inside:

```text
data/Digital Marketing Campaign Dataset/
```

Example file:

```text
digital_marketing_campaign_dataset.csv
```

Expected target column:

```text
Conversion
```

---

# Run Project

Use this command:

```bash
python main_llm.py --auto --dataset "data/Digital Marketing Campaign Dataset/digital_marketing_campaign_dataset.csv" --planner-llm ollama --planner-model qwen3:4b --decision-llm ollama --decision-model qwen3:4b
```

---

# Workflow Pipeline

The system runs this workflow:

```text
1. load_and_inspect_data
2. preprocess_data
3. analyze_data
4. generate_recommendations
```

---

# Supported Analysis Tasks

## Classification

Used for:

- conversion prediction
- campaign success prediction
- customer response prediction

## Regression

Used for:

- conversion rate prediction
- CTR prediction
- campaign performance score prediction

## Anomaly Detection

Used for:

- unusual campaign behavior
- abnormal customer engagement
- suspicious marketing performance

---

# Machine Learning Models

## Classification Models

- RandomForestClassifier
- GradientBoostingClassifier
- LogisticRegression
- SVC

## Regression Models

- RandomForestRegressor
- GradientBoostingRegressor
- LinearRegression
- Ridge
- Lasso
- SVR

---

# Recommendation Output

The system can generate recommendations such as:

- revise ad creatives
- optimize campaign targeting
- increase budget for high-performing channels
- reduce budget for weak campaigns
- improve landing page strategy
- retarget high-intent users
- improve email CTA
- prioritize retention campaigns
- optimize audience segmentation

---

# Output Files

Generated files are saved in:

```text
logs/
```

Possible outputs:

```text
digital_marketing_workflow_report_*.json
digital_marketing_detailed_results_*.json
publication_snapshot_*.json
publication_snapshot_*_recommendations.csv
```

---

# Research Focus

This project can support research on:

- AI agent-based decision support systems
- local LLM orchestration
- digital marketing campaign optimization
- advertising decision recommendation
- conversion prediction
- human-in-the-loop AI
- multi-agent analytics systems
- structured output reliability in local LLMs

---

# Known Challenges

Local LLMs may produce:

- verbose reasoning before JSON
- invalid JSON formatting
- hallucinated workflow summaries
- inconsistent planner decisions

This project addresses these using:

- robust JSON parsing
- deterministic fallback planning
- retry logic
- tool validation
- human-in-the-loop review

---

# Example Research Title

```text
AI Agent-Based Decision Support System for Digital Marketing Campaign Optimization Using Local Large Language Models
```

---

# Notes

If the workflow only shows:

```text
Steps: 1/1 succeeded
```

then the workflow has not fully executed.

A successful run should show:

```text
Executing tool: load_and_inspect_data
Executing tool: preprocess_data
Executing tool: analyze_data
Executing tool: generate_recommendations
```

---

# Author

Adapted and extended for Digital Marketing and Advertising AI Agent research.

Original repository:

```text
https://github.com/tamoraji/smart_manufacturing_mas_code
```
---

## Agentic AI Evaluation Upgrade

This version includes a dedicated **Framework Evaluasi Agentic AI untuk Analisa dan Pengambilan Keputusan Digital Marketing**. In addition to the original ML/model benchmark pipeline, the system now evaluates the agent workflow itself through:

- agentic AI metrics;
- reasoning trajectory logging with `thought/action/observation/reflection`;
- planning evaluation;
- tool selection evaluation;
- robustness testing;
- governance and safety evaluation;
- benchmark scoring;
- multi-agent evaluation;
- audit logging;
- adaptive workflow fallback;
- visualization/reporting artifacts.

Run offline/mock mode:

```bash
python main_llm.py --dataset data.csv --planner-llm mock --decision-llm mock --auto
```

Run Qwen workflow:

```bash
python main_comparison.py --dataset data.csv --qwen-model qwen3:4b --auto
```

New outputs are saved under `logs/`, including `agentic_trajectory_*.jsonl` and `agentic_evaluation_report_*` files.

See `documentation/agentic_evaluation_framework.md` for the full architecture notes.
