# AI Native Data Analytics Platform

Production-grade enterprise data analytics platform with **Medallion Architecture**, **Hugging Face AI**, **FAISS vector search**, autonomous AI agents, and natural language analytics.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT FRONTEND                           │
│  Home │ Upload │ Profiling │ Quality │ Schema │ Lineage │ Dashboard  │
│  AI Insights │ Chat │ Copilot │ Features │ Catalog │ Observability  │
├──────────────────────────────────────────────────────────────────────┤
│                      AI AGENTS + COPILOT                            │
│  Understanding │ Dashboard Builder │ Insight │ Pipeline │ Debug      │
│                    AI Data Copilot                                   │
├──────────────────────────────────────────────────────────────────────┤
│                  AI ENGINE (HUGGING FACE)                            │
│  Insights │ Chat │ SQL Generator │ Vector Search (FAISS)             │
├──────────────────────────────────────────────────────────────────────┤
│                  MEDALLION ARCHITECTURE                              │
│   🥉 Bronze (Raw)  →  🥈 Silver (Clean)  →  🥇 Gold (Analytics)     │
├──────────────────────────────────────────────────────────────────────┤
│                   DATA PLATFORM SERVICES                             │
│  Schema Registry │ Lineage │ Metadata │ Features │ Governance │ Obs  │
├──────────────────────────────────────────────────────────────────────┤
│                    DATA INGESTION                                    │
│              CSV  │  Excel  │  Validation                            │
└──────────────────────────────────────────────────────────────────────┘
```

## Medallion Architecture

| Layer | Purpose | Operations |
|-------|---------|------------|
| **Bronze** | Raw ingested data | No modifications, full metadata capture |
| **Silver** | Cleaned and validated data | Dedup, fill nulls, standardize names, normalize types |
| **Gold** | Business analytics tables & KPIs | Aggregations, derived metrics, business analytics |

---

## Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Data Ingestion** | CSV/Excel upload with validation and metadata capture |
| 2 | **Data Profiling** | Types, missing values, distributions, summary statistics |
| 3 | **Data Quality** | Null/duplicate/outlier checks, scoring (0–100), alerts |
| 4 | **Schema Registry** | Versioned schema detection and storage |
| 5 | **Schema Evolution** | Detects added/removed columns, type changes, renames |
| 6 | **Data Lineage** | End-to-end pipeline tracking with visual graph |
| 7 | **Transformations** | Modular cleaning, feature engineering, derived columns |
| 8 | **Analytics Dashboard** | KPI cards, auto-generated charts, custom chart builder |
| 9 | **AI Insights** | HuggingFace-powered insights with fallback analysis |
| 10 | **Chat with Data** | Natural language queries converted to pandas operations |
| 11 | **Auto SQL Generator** | NL → SQL + pandas code with execution |
| 12 | **AI Data Copilot** | Conversational assistant for the entire platform |
| 13 | **Vector Search** | FAISS semantic search over dataset metadata |
| 14 | **Feature Store** | Reusable derived metrics (CLV, frequency, z-scores) |
| 15 | **Metadata Catalog** | Central registry with semantic layer |
| 16 | **Data Governance** | Ownership, classification, access logs |
| 17 | **Data Observability** | Pipeline monitoring, quality trends, event tracking |
| 18 | **5 AI Agents** | Understanding, Dashboard, Insight, Pipeline, Debug |

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python |
| Frontend | Streamlit (sidebar page navigation) |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| AI/LLM | Hugging Face Inference API |
| Vector Search | FAISS + Sentence Transformers |
| Metadata | JSON registry |
| Pipeline | Python pipeline modules |

---

## Project Structure

```
ai_data_platform/
├── app.py                              # Main Streamlit app (14 pages)
├── requirements.txt
├── README.md
├── .env                                # HuggingFace API token
├── ingestion/data_loader.py            # Data Ingestion Engine
├── bronze/bronze_storage.py            # Bronze Layer
├── profiling/data_profiler.py          # Data Profiling Engine
├── quality/data_quality_engine.py      # Data Quality Framework
├── silver/data_cleaner.py              # Silver Layer
├── schema/
│   ├── schema_registry.py             # Schema Registry
│   └── schema_evolution.py            # Schema Evolution
├── lineage/lineage_tracker.py          # Data Lineage
├── metadata/metadata_store.py          # Metadata Catalog
├── transformations/transform_pipeline.py
├── gold/analytics_tables.py            # Gold Layer
├── analytics/
│   ├── kpi_engine.py                  # KPI Engine
│   └── chart_generator.py            # Visualization Engine
├── ai/
│   ├── llm_client.py                  # Shared HuggingFace client
│   ├── insights_engine.py             # AI Insights
│   ├── chat_engine.py                 # Chat with Data
│   └── sql_generator.py              # Auto SQL Generator
├── ai_agents/
│   ├── data_understanding_agent.py
│   ├── dashboard_builder_agent.py
│   ├── insight_agent.py
│   ├── pipeline_agent.py
│   ├── debug_agent.py
│   └── copilot_agent.py              # AI Data Copilot
├── vector_search/
│   └── embedding_engine.py           # FAISS Vector Search
├── feature_store/feature_registry.py
├── governance/data_governance.py
├── observability/monitoring_engine.py
└── data/                              # Runtime storage (auto-created)
```

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Hugging Face (for AI features)

Get a free API token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Add it to `.env`:

```
HF_API_TOKEN=hf_your_token_here
```

### 3. Run the Platform

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

> **Note:** The platform works fully without a HuggingFace token — all data processing, profiling, quality checks, visualizations, and KPIs work independently. AI features (insights, chat, copilot, SQL generation) fall back to statistical analysis when no token is set.

---

## Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Welcome screen with feature overview |
| 📤 Upload Dataset | CSV/Excel upload with pipeline execution |
| 📋 Data Profiling | Column types, missing values, distributions |
| ✅ Data Quality | Quality score, alerts, column scores |
| 📜 Schema Registry | Versioned schema detection |
| 🔄 Schema Evolution | Cross-version schema changes |
| 🔗 Data Lineage | Visual pipeline tracking |
| ⚙️ Transformations | Cleaning and feature engineering details |
| 📊 Analytics Dashboard | KPIs, gold tables, charts, custom builder |
| 💡 AI Insights | AI insights, anomalies, patterns, agents |
| 💬 Chat With Data | NL chat, auto SQL, vector search |
| 🤖 AI Copilot | Conversational data assistant |
| 📦 Feature Store | Derived metrics registry |
| 🗂️ Metadata Catalog | Schema, governance, semantic layer |
| 📡 Observability | Pipeline monitoring, quality trends |

---

Built with Streamlit + Hugging Face + FAISS | Medallion Architecture | Enterprise Data Platform
