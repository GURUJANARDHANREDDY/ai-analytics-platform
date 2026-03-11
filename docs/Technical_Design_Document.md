# AI-Native Data Analytics Platform — Technical Design Document

## 1. Executive Summary

This platform is an AI-powered data analytics solution that replaces traditional drag-and-drop BI tools with intelligent, automated dashboard generation. Users upload any CSV or Excel dataset and receive a fully interactive analytics dashboard — either designed by an AI (Llama 3.1 LLM) or auto-generated using rule-based heuristics.

The platform supports natural language querying, auto SQL generation, data governance, and enterprise-grade data pipeline architecture (Medallion pattern).

**Live Demo:** [Streamlit Cloud URL]
**Repository:** https://github.com/GURUJANARDHANREDDY/ai-analytics-platform

---

## 2. Problem Statement

Traditional BI tools like Tableau and Power BI require:
- Manual chart selection and layout design
- Expertise in knowing which visualization suits which data type
- Time-consuming dashboard building (hours per dashboard)
- Separate SQL knowledge for data querying

**This platform solves these by:**
- Letting AI decide the optimal dashboard layout, KPIs, and chart types
- Generating dashboards in seconds, not hours
- Enabling plain English questions instead of SQL
- Automating data quality checks and governance

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │
│  │AI Dash   │ │Auto Dash │ │Ask AI    │ │Governance │  │
│  │(LLM)     │ │(Rules)   │ │(NL→SQL) │ │(Quality)  │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └─────┬─────┘  │
│       │             │            │              │        │
├───────┼─────────────┼────────────┼──────────────┼────────┤
│       ▼             ▼            ▼              ▼        │
│              BACKEND PROCESSING LAYER                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐  │
│  │Data Loader   │ │Data Profiler │ │Column Classifier │  │
│  │(CSV/Excel)   │ │(Statistics)  │ │(ID/Measure/Dim)  │  │
│  └──────────────┘ └──────────────┘ └──────────────────┘  │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                    AI / LLM LAYER                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐  │
│  │Dashboard     │ │Chat Engine   │ │SQL Generator     │  │
│  │Designer      │ │(Q&A)         │ │(NL → Code)       │  │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────────┘  │
│         └────────────────┼────────────────┘              │
│                          ▼                               │
│              Hugging Face API (Llama 3.1 8B)             │
└──────────────────────────────────────────────────────────┘
```

---

## 4. Core Components

### 4.1 Data Ingestion Layer
- **Supported formats:** CSV, XLSX, XLS
- **Encoding detection:** Auto-detects UTF-8, UTF-8-BOM, UTF-16, CP1252, Latin-1
- **Parsing strategy:** PyArrow (fast) → Pandas fallback with multiple encodings
- **Date detection:** Auto-parses object columns that contain >50% valid dates
- **Validation:** File size limit (100MB), extension check, empty file check

### 4.2 Smart Column Classification
This is the intelligence layer that makes auto-generation possible.

**Classification Categories:**
| Category | Detection Logic | Example |
|----------|----------------|---------|
| Identifier | Column name contains "id", "key", "code" OR high cardinality numeric | Customer_ID, Order_No |
| Measure | Numeric column that is NOT an identifier | Revenue, Quantity, Profit |
| Dimension | Categorical/object column with reasonable cardinality | Region, Category, Segment |
| Datetime | Datetime dtype or auto-parsed date column | Order_Date, Created_At |

**Why this matters:** Without smart classification, the platform would show "Sum of Customer_ID = 1,450,230" as a KPI — which is meaningless. The classifier ensures only true measures are used for aggregations.

### 4.3 AI Dashboard Designer (LLM-Powered)
**How it works:**

1. User clicks "Generate AI Dashboard"
2. System sends to Llama 3.1:
   - Column names, data types, unique counts
   - Sample values (3 per column)
   - Pre-classified measures, dimensions, datetime columns
   - Row count and shape
3. LLM returns a JSON specification:
   ```json
   {
     "title": "MTA Employee Payroll Analysis",
     "subtitle": "Compensation trends across agencies",
     "kpis": [{"column": "Regular Gross Paid", "agg": "sum", "label": "Total Payroll"}],
     "charts": [{"type": "bar", "x": "Agency Name", "y": "Regular Gross Paid", "agg": "sum", "top_n": 10, "reason": "Compare payroll across agencies"}],
     "insights": ["Top 5 agencies account for 62% of total payroll"]
   }
   ```
4. Platform parses and validates the JSON (checks columns exist, chart types are valid)
5. Renders the dashboard using Plotly based on AI specifications
6. Each chart shows the AI's reasoning

**Fallback:** If LLM fails (timeout, invalid JSON, API key missing), the platform falls back to rule-based auto-generation.

### 4.4 Auto Dashboard (Rule-Based)
Generates charts using deterministic rules:

| Data Pattern | Chart Type | Logic |
|-------------|------------|-------|
| 1 dimension + 1 measure | Bar chart | Group by dimension, aggregate measure |
| Datetime + measure | Line chart | Group by month, show trend |
| 2 dimensions + measure | Grouped bar | Cross-tabulation |
| 2 measures | Scatter plot | Correlation analysis |
| 1 dimension (low cardinality) | Pie/Donut | Distribution |
| 1 dimension + 1 measure | Treemap | Proportional composition |
| 1 dimension + 1 measure | Box plot | Distribution spread |
| 3+ measures | Heatmap | Correlation matrix |

### 4.5 Natural Language to SQL/Pandas
**Flow:**
1. User types: "Show top 10 products by revenue"
2. LLM generates:
   - SQL: `SELECT Product, SUM(Revenue) FROM dataset GROUP BY Product ORDER BY SUM(Revenue) DESC LIMIT 10`
   - Pandas: `result = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False).head(10)`
3. Code cleaning:
   - Strips `import` statements (security)
   - Strips `print()` calls (causes errors in sandbox)
   - Removes comments and empty lines
4. Safe execution:
   - Restricted `__builtins__` (only `len`, `str`, `int`, `float`, `sorted`, `round`, etc.)
   - `print` replaced with no-op lambda
   - Only `pd`, `np`, and `df` available in namespace
5. Result displayed as styled HTML table + auto-generated chart

### 4.6 Data Governance
Automated quality assessment with no manual configuration:

- **Completeness:** (1 - missing_cells / total_cells) × 100
- **Uniqueness:** (1 - duplicate_rows / total_rows) × 100
- **Consistency:** Detects case/whitespace inconsistencies in text columns
- **Validity:** Outlier detection using IQR method (Q1 - 1.5×IQR, Q3 + 1.5×IQR)
- **Overall Score:** Weighted average (35% completeness, 25% uniqueness, 20% consistency, 20% validity)

Validation rules auto-generated per column:
- Null check (PASS/WARN/FAIL based on % missing)
- Duplicate row check
- Outlier check per numeric column
- Consistency check per text column

---

## 5. Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Frontend | Streamlit | Rapid prototyping, Python-native, built-in widgets |
| Charts | Plotly | Interactive, dark theme support, 20+ chart types |
| Data Processing | Pandas + PyArrow | PyArrow for fast CSV parsing, Pandas for analysis |
| LLM | Meta Llama 3.1 8B | Free via Hugging Face, good at structured output |
| LLM API | Hugging Face Inference API | Free tier, OpenAI-compatible endpoint |
| ML | scikit-learn | Feature importance (Random Forest) |
| Deployment | Streamlit Community Cloud | Free hosting, GitHub integration |

---

## 6. Key Design Decisions

### 6.1 Why LLM for Dashboard Design?
Traditional auto-generation uses fixed rules. The LLM adds contextual understanding:
- It recognizes "Regular Gross Paid" is a salary column and labels the KPI "Total Payroll" (not "Sum of Regular Gross Paid")
- It understands domain context — payroll data gets different charts than sales data
- It provides insights that rules cannot generate

### 6.2 Why Two Dashboard Modes?
- **AI Dashboard:** Better titles, smarter KPI selection, contextual insights — but requires API call (~10 seconds)
- **Auto Dashboard:** Instant, no API dependency, deterministic — but generic labels and fixed logic
- Users can compare both approaches side by side

### 6.3 Why HTML Tables Instead of st.dataframe?
Streamlit's built-in `st.dataframe` uses a canvas-based renderer (Glide Data Grid) where text color cannot be controlled via CSS. On dark themes, text becomes invisible. Styled HTML tables give full control over text visibility.

### 6.4 Why Safe Code Execution?
AI-generated code can contain:
- `import os; os.system("rm -rf /")` — dangerous
- `print(result)` — causes "print not defined" in sandbox
- Multi-line lambdas — execution errors

Solution: Strip imports, strip print, restrict builtins, execute in isolated namespace.

---

## 7. Project Structure

```
ai_analytical_platform/
├── app.py                      # Entry point (launcher)
├── enterprise_app.py           # Enterprise Medallion Architecture app
├── frontend/
│   ├── app.py                  # Main Streamlit UI (tabs, layout, CSS)
│   ├── tableau_dashboard.py    # AI + Auto dashboard rendering
│   ├── charts_ui.py            # Auto chart generation & custom builder
│   ├── chat_ui.py              # Chat interface component
│   ├── dashboard.py            # Data overview rendering
│   └── enterprise_ui.py        # Governance & observability UI
├── ai/
│   ├── llm_client.py           # Hugging Face API client
│   ├── dashboard_designer.py   # LLM-based dashboard design
│   └── sql_generator.py        # NL to SQL/Pandas
├── backend/
│   ├── ai_chat_engine.py       # Q&A engine with safe code execution
│   ├── data_loader.py          # CSV/Excel ingestion
│   ├── data_profiler.py        # Statistical profiling
│   ├── narrative_engine.py     # Rule-based insights generation
│   └── config.py               # Settings from env/secrets
├── utils/
│   ├── column_classifier.py    # Smart column type detection
│   ├── file_utils.py           # File validation & hashing
│   └── logging_utils.py        # Logging configuration
├── bronze/                     # Bronze layer (raw storage)
├── silver/                     # Silver layer (cleaning)
├── gold/                       # Gold layer (analytics tables)
├── schema/                     # Schema registry & evolution
├── lineage/                    # Data lineage tracking
├── .streamlit/config.toml      # Dark theme configuration
└── requirements.txt            # Python dependencies
```

---

## 8. Data Flow

```
Upload (CSV/Excel)
    │
    ▼
Encoding Detection → Multi-strategy parsing
    │
    ▼
Auto Date Parsing → Column Classification
    │                (ID / Measure / Dimension / DateTime)
    ▼
Data Profiling → Statistics, distributions, anomalies
    │
    ├──────────────────┬────────────────────┐
    ▼                  ▼                    ▼
AI Dashboard       Auto Dashboard      Data Governance
(LLM designs)      (Rules pick)        (Quality scoring)
    │                  │                    │
    ▼                  ▼                    ▼
Interactive Plotly Charts + Filters + KPI Cards
    │
    ▼
AI Q&A (Natural language → SQL → Execute → Visualize)
```

---

## 9. Enterprise Architecture (Medallion Pattern)

The separate enterprise app implements a production data pipeline:

| Layer | Purpose | Implementation |
|-------|---------|---------------|
| Bronze | Raw data storage | Immutable copy of uploaded file with metadata |
| Silver | Cleaned data | Deduplication, null handling, type inference, normalization |
| Gold | Analytics tables | Pre-aggregated tables by dimension, time series |
| Feature Store | ML features | Auto-computed features (ratios, z-scores, percentiles) |
| Schema Registry | Schema tracking | Version history, drift detection |
| Lineage Tracker | Pipeline audit | Event log for every transformation stage |

---

## 10. Security Considerations

| Risk | Mitigation |
|------|-----------|
| Code injection via AI | Restricted builtins, stripped imports, sandboxed exec |
| API key exposure | .env excluded from git, st.secrets for cloud |
| Large file DoS | 100MB file size limit |
| XSS via data values | HTML table rendering escapes values |

---

## 11. Limitations & Future Scope

### Current Limitations
- Single file analysis (no multi-table joins)
- No database connections (CSV/Excel only)
- LLM can produce invalid JSON requiring fallback
- No real-time data / streaming
- Single-user (no auth / multi-tenant)
- No dashboard persistence (resets on refresh)

### Future Enhancements
- Database connectors (PostgreSQL, MySQL, BigQuery)
- Fine-tuned LLM for dashboard design (better accuracy)
- Dashboard templates and saving
- Scheduled data refresh
- Multi-user with authentication
- Cross-chart filtering (click bar → filter all charts)
- Export to PDF / PowerPoint

---

## 12. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Set API token
echo "HF_API_TOKEN=your_token_here" > .env

# Run main app (AI dashboards)
streamlit run app.py

# Run enterprise app (Medallion architecture)
streamlit run enterprise_app.py --server.port 8502
```

---

## 13. Key Metrics

- **Dashboard generation time:** <1 second (auto) / ~10 seconds (AI)
- **Supported chart types:** 11 (bar, line, scatter, pie, donut, treemap, box, histogram, heatmap, grouped bar, horizontal bar)
- **Max file size:** 100 MB
- **Code size:** ~13,000 lines across 93 files
- **Dependencies:** 10 Python packages
