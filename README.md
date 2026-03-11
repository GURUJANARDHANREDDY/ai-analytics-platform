# AI-Native Data Analytics Platform

An AI-powered analytics platform that generates interactive dashboards from any dataset. Upload a CSV or Excel file — the AI analyzes your data, designs the optimal dashboard layout, picks the right KPIs and chart types, and builds a fully interactive analytics experience.

## Features

### AI-Designed Dashboards
- Click "Generate AI Dashboard" — Llama 3.1 analyzes your columns, data types, and distributions
- AI picks the title, KPIs, chart types, column pairings, and explains why each chart was chosen
- Click "Redesign" for a completely new AI-designed layout

### Auto-Generated Dashboards
- Rule-based chart selection using smart column classification
- Instant generation with no API call required
- Compare AI vs Auto approaches side by side

### Natural Language Querying
- Ask any question in plain English: "What are the top 10 products by revenue?"
- AI generates SQL + Pandas code and executes it live
- Auto-generates charts from query results

### Data Governance
- Quality scoring: completeness, uniqueness, consistency, validity
- Automated validation rules per column (PASS/WARN/FAIL)
- Column health reports and distribution analysis

### Enterprise Architecture (Separate App)
- Medallion Architecture: Bronze → Silver → Gold layers
- Schema registry with version history and drift detection
- Data lineage tracking across pipeline stages
- Feature store with auto-computed features

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit, Plotly |
| Data | Pandas, PyArrow, NumPy |
| AI/LLM | Meta Llama 3.1 8B (Hugging Face) |
| ML | scikit-learn |
| Deployment | Streamlit Community Cloud |

## Quick Start

```bash
# Clone
git clone https://github.com/GURUJANARDHANREDDY/ai-analytics-platform.git
cd ai-analytics-platform

# Install
pip install -r requirements.txt

# Configure (get token from https://huggingface.co/settings/tokens)
echo "HF_API_TOKEN=your_token" > .env

# Run
streamlit run app.py
```

## Architecture

```
Upload CSV/Excel
    → Encoding detection & multi-strategy parsing
    → Smart column classification (ID / Measure / Dimension / DateTime)
    → Data profiling (statistics, distributions, anomalies)
    → AI Dashboard Design (LLM) OR Auto Dashboard (Rules)
    → Interactive Plotly charts + filters + KPI cards
    → AI Q&A (Natural language → SQL → Execute → Visualize)
```

## Two Apps

| App | Command | Port | Purpose |
|-----|---------|------|---------|
| AI Analytics | `streamlit run app.py` | 8501 | Dashboards, AI chat, SQL, governance |
| Enterprise | `streamlit run enterprise_app.py --server.port 8502` | 8502 | Medallion pipeline, schema, lineage |

## Documentation

See [Technical Design Document](docs/Technical_Design_Document.md) for detailed architecture, design decisions, and implementation details.
