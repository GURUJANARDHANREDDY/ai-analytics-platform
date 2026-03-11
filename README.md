# AI-Powered Dashboard Platform

Upload any CSV or Excel file — the AI analyzes your data, designs the optimal dashboard, picks the right KPIs and chart types, and builds an interactive analytics experience in seconds.

## Features

### AI-Designed Dashboards
- Click "Generate AI Dashboard" — Llama 3.1 analyzes your columns, types, and distributions
- AI picks the title, KPIs, chart types, column pairings, and explains why each chart was chosen
- Click "Redesign" for a completely new AI-designed layout

### Auto-Generated Dashboards
- Rule-based chart selection using smart column classification
- Instant generation — no API call needed
- Compare AI vs Auto approaches side by side

### Natural Language Querying
- Ask any question in plain English: "What are the top 10 products by revenue?"
- AI generates SQL + Pandas code and executes it live
- Auto-generates charts from query results

### Data Governance
- Quality scoring: completeness, uniqueness, consistency, validity
- Automated validation rules per column (PASS/WARN/FAIL)
- Column health reports and distribution analysis

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
pip install -r requirements.txt
echo "HF_API_TOKEN=your_token" > .env
streamlit run app.py
```

## Documentation

See [Technical Design Document](docs/Technical_Design_Document.md) for architecture, approach, implementation details, and design decisions.
