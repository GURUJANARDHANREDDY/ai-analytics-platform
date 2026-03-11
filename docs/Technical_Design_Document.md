# AI-Powered Dashboard Platform — Technical Design Document

## 1. Executive Summary

This platform is an AI-powered data analytics solution that generates interactive dashboards from any dataset. Users upload a CSV or Excel file — the AI (Llama 3.1) analyzes the data structure, designs the optimal dashboard layout, picks the right KPIs and chart types, and builds a fully interactive analytics experience in seconds.

It also supports natural language querying (Ask AI), auto SQL generation, and automated data governance.

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
- Automating data quality checks

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                     │
│  ┌───────────┐ ┌───────────┐ ┌──────────┐ ┌──────────┐  │
│  │AI Dash    │ │Auto Dash  │ │Ask AI    │ │Governance│  │
│  │(LLM)     │ │(Rules)    │ │(NL→SQL)  │ │(Quality) │  │
│  └─────┬─────┘ └─────┬─────┘ └────┬─────┘ └────┬─────┘  │
│        │              │            │             │        │
├────────┼──────────────┼────────────┼─────────────┼────────┤
│        ▼              ▼            ▼             ▼        │
│               BACKEND PROCESSING LAYER                    │
│  ┌──────────────┐ ┌──────────────┐ ┌───────────────────┐  │
│  │Data Loader   │ │Data Profiler │ │Column Classifier  │  │
│  │(CSV/Excel)   │ │(Statistics)  │ │(ID/Measure/Dim)   │  │
│  └──────────────┘ └──────────────┘ └───────────────────┘  │
│                                                           │
├───────────────────────────────────────────────────────────┤
│                     AI / LLM LAYER                        │
│  ┌───────────────┐ ┌──────────────┐ ┌─────────────────┐  │
│  │Dashboard      │ │Chat Engine   │ │SQL Generator    │  │
│  │Designer       │ │(Q&A)         │ │(NL → Code)      │  │
│  └───────┬───────┘ └──────┬───────┘ └──────┬──────────┘  │
│          └────────────────┼────────────────┘              │
│                           ▼                               │
│               Hugging Face API (Llama 3.1 8B)             │
└───────────────────────────────────────────────────────────┘
```

---

## 4. Approach

### Step 1: Data Ingestion
When a user uploads a file, the platform:
- Detects encoding (UTF-8, BOM, CP1252, Latin-1) automatically
- Tries PyArrow first (fast), falls back to Pandas with multiple encodings
- Supports CSV, XLSX, and XLS formats
- Auto-parses date columns (if >50% of values are valid dates)
- Validates file size (max 100MB) and format

### Step 2: Smart Column Classification
This is the core intelligence that makes dashboard generation possible.

| Category | How It's Detected | Example |
|----------|------------------|---------|
| Identifier | Name contains "id", "key", "code" OR high-cardinality numeric | Customer_ID, Order_No |
| Measure | Numeric column that is NOT an identifier | Revenue, Quantity, Profit |
| Dimension | Categorical/text column with reasonable cardinality | Region, Category, Segment |
| Datetime | Datetime dtype or auto-parsed date column | Order_Date, Start_Date |

**Why this matters:** Without this, the platform would show "Sum of Customer_ID = 1,450,230" as a KPI — completely meaningless. The classifier ensures only real measures get aggregated.

### Step 3: Dashboard Generation (Two Approaches)

**Approach A — AI-Designed (LLM)**

1. User clicks "Generate AI Dashboard"
2. Platform sends column metadata to Llama 3.1:
   - Column names, data types, unique counts, sample values
   - Pre-classified measures, dimensions, datetime columns
3. LLM returns a JSON specification:
```json
{
  "title": "MTA Employee Payroll Analysis",
  "subtitle": "Compensation trends across agencies",
  "kpis": [
    {"column": "Regular Gross Paid", "agg": "sum", "label": "Total Payroll"}
  ],
  "charts": [
    {
      "type": "bar",
      "x": "Agency Name",
      "y": "Regular Gross Paid",
      "agg": "sum",
      "top_n": 10,
      "reason": "Compare payroll spend across agencies"
    }
  ],
  "insights": ["Top 5 agencies account for 62% of total payroll"]
}
```
4. Platform validates the JSON (checks columns exist, chart types are supported)
5. Renders the dashboard using Plotly
6. Each chart shows the AI's reasoning for choosing it
7. User can click "Redesign" for a completely new layout

**Approach B — Auto-Generated (Rule-Based)**

Instant dashboard using deterministic rules — no LLM call needed:

| Data Pattern | Chart Type | Logic |
|-------------|------------|-------|
| 1 dimension + 1 measure | Bar chart | Group by dimension, sum measure |
| Datetime + measure | Line chart | Group by month, show trend |
| 2 dimensions + measure | Grouped bar | Cross-tabulation |
| 2 measures | Scatter plot | Correlation analysis |
| 1 dimension (low cardinality) | Pie/Donut | Distribution |
| 1 dimension + 1 measure | Treemap | Proportional composition |
| 1 dimension + 1 measure | Box plot | Distribution spread |
| 3+ measures | Heatmap | Correlation matrix |

**Why both?** AI Dashboard gives smarter titles, contextual KPIs, and insights — but takes ~10 seconds. Auto Dashboard is instant and works without an API key. Users can compare both side by side.

### Step 4: Interactive Features
- **Filters:** Select any dimension to filter — all KPIs and charts update instantly
- **Measure selector:** Switch which numeric column drives the charts
- **KPI cards:** Show totals, averages, and delta vs full dataset when filtered
- **Ranking tables:** Top N items by dimension with sum, mean, and count

### Step 5: Natural Language Querying (Ask AI)
Users type plain English questions:

```
User: "What are the top 10 products by revenue?"
```

**Flow:**
1. LLM generates SQL + Pandas code
2. Code cleaning removes `import`, `print()`, comments (security)
3. Safe execution in sandboxed namespace (restricted builtins)
4. Result displayed as HTML table + auto-generated chart

### Step 6: Data Governance
Automated quality assessment — no configuration needed:

| Metric | Formula | Weight |
|--------|---------|--------|
| Completeness | (1 - missing_cells / total_cells) × 100 | 35% |
| Uniqueness | (1 - duplicate_rows / total_rows) × 100 | 25% |
| Consistency | Detects case/whitespace issues in text columns | 20% |
| Validity | Outlier count using IQR method | 20% |

Plus auto-generated validation rules per column (PASS / WARN / FAIL).

---

## 5. Implementation Details

### 5.1 AI Dashboard Designer (`ai/dashboard_designer.py`)
- Constructs a prompt with column metadata and sample values
- Sends to Hugging Face API (OpenAI-compatible endpoint)
- Parses JSON response with regex fallback for malformed output
- Validates every column reference against actual DataFrame
- Filters out invalid chart specs before rendering

### 5.2 Chart Renderer (`frontend/tableau_dashboard.py`)
- Takes AI's JSON chart spec and maps to Plotly functions
- Supports 11 chart types: bar, horizontal_bar, grouped_bar, line, scatter, pie, donut, treemap, box, histogram, heatmap
- Each chart type has aggregation logic (sum, mean, count)
- Handles edge cases: mixed-type columns, high cardinality, null values

### 5.3 Safe Code Execution (`backend/ai_chat_engine.py`)
AI-generated code is dangerous by default. The platform:
- Strips all `import` and `from` statements
- Strips `print()` calls (causes sandbox errors)
- Restricts `__builtins__` to safe functions only: `len`, `str`, `int`, `float`, `sorted`, `round`, `abs`, `sum`, `min`, `max`
- Injects `print = lambda *a, **k: None` as safety net
- Only exposes `pd`, `np`, and `df` in execution namespace

### 5.4 Column Classifier (`utils/column_classifier.py`)
Detection logic for each column:
1. Check if name matches ID patterns (id, key, code, number, no, index)
2. Check if numeric but high cardinality (>80% unique) — likely an ID
3. Check if numeric with low cardinality — could be categorical
4. Remaining numerics → measures
5. Object/category dtype → dimensions
6. Datetime dtype → datetime

### 5.5 Data Loader (`backend/data_loader.py`)
Multi-strategy parsing:
1. Detect encoding from BOM markers and byte analysis
2. Try PyArrow CSV reader (fastest, UTF-8 only)
3. Fall back to Pandas with detected encoding
4. Try CP1252, Latin-1 as last resort
5. For Excel: try openpyxl engine, fall back to xlrd

---

## 6. Technology Stack

| Component | Technology | Why Chosen |
|-----------|-----------|------------|
| Frontend | Streamlit | Python-native, rapid prototyping, built-in widgets |
| Charts | Plotly | Interactive, dark theme, 20+ chart types, hover tooltips |
| Data Processing | Pandas + PyArrow | PyArrow for fast parsing, Pandas for analysis |
| LLM | Meta Llama 3.1 8B | Free via Hugging Face, good at structured JSON output |
| LLM API | Hugging Face Inference API | Free tier, OpenAI-compatible endpoint |
| ML | scikit-learn | Feature importance via Random Forest |
| Deployment | Streamlit Community Cloud | Free, GitHub integration, secrets management |

---

## 7. Key Design Decisions

### Why LLM for Dashboard Design?
Rule-based auto-generation picks charts by data type alone. The LLM adds contextual understanding:
- Recognizes "Regular Gross Paid" is salary and labels it "Total Payroll" (not "Sum of Regular Gross Paid")
- Understands domain context — payroll data gets different charts than e-commerce data
- Generates insights that rules can never produce

### Why HTML Tables Instead of st.dataframe?
Streamlit's `st.dataframe` uses a canvas-based renderer (Glide Data Grid). On dark themes, the canvas paints text in dark colors that become invisible. CSS cannot override canvas rendering. HTML tables give full control over text color and styling.

### Why Two Dashboard Modes?
- Users without an API key still get a working dashboard (Auto mode)
- Analysts can compare AI vs rule-based approaches
- AI mode demonstrates the value of LLM-powered analytics

---

## 8. Project Structure

```
ai_analytical_platform/
├── app.py                       # Entry point
├── frontend/
│   ├── app.py                   # Main UI (tabs, layout, CSS)
│   ├── tableau_dashboard.py     # AI + Auto dashboard rendering
│   ├── charts_ui.py             # Chart generation & custom builder
│   ├── chat_ui.py               # Chat interface
│   └── dashboard.py             # Data overview
├── ai/
│   ├── llm_client.py            # Hugging Face API client
│   ├── dashboard_designer.py    # LLM-based dashboard design
│   └── sql_generator.py         # NL to SQL/Pandas
├── backend/
│   ├── ai_chat_engine.py        # Q&A with safe code execution
│   ├── data_loader.py           # CSV/Excel ingestion
│   ├── data_profiler.py         # Statistical profiling
│   ├── narrative_engine.py      # Rule-based insights
│   └── config.py                # Settings from env/secrets
├── utils/
│   ├── column_classifier.py     # Smart column type detection
│   ├── file_utils.py            # File validation & hashing
│   └── logging_utils.py         # Logging config
├── .streamlit/config.toml       # Dark theme
└── requirements.txt             # Dependencies
```

---

## 9. Data Flow

```
Upload CSV / Excel
       │
       ▼
Encoding Detection → Multi-strategy Parsing
       │
       ▼
Auto Date Parsing → Smart Column Classification
       │              (ID / Measure / Dimension / DateTime)
       ▼
Data Profiling → Statistics, distributions, anomalies
       │
       ├─────────────────────┬────────────────────┐
       ▼                     ▼                    ▼
 AI Dashboard          Auto Dashboard       Data Governance
 (LLM designs          (Rules pick           (Quality scoring,
  KPIs, charts,         charts by             validation rules)
  insights)             data type)
       │                     │                    │
       ▼                     ▼                    ▼
 Interactive Plotly Charts + Filters + KPI Cards
       │
       ▼
 AI Q&A (English → SQL/Pandas → Execute → Visualize)
```

---

## 10. Security

| Risk | Mitigation |
|------|-----------|
| Code injection via AI-generated code | Restricted builtins, stripped imports, sandboxed exec |
| API key exposure | .env excluded from git, st.secrets for cloud deployment |
| Large file upload attack | 100MB file size limit, extension validation |
| XSS via data values in tables | HTML escaping in table rendering |

---

## 11. Limitations

- Single file analysis only (no multi-table joins or database connections)
- LLM can return invalid JSON — falls back to auto-generation
- No real-time data streaming (static file analysis)
- Single-user (no authentication or multi-tenant support)
- No dashboard saving or export (resets on browser refresh)
- AI dashboard takes ~10 seconds (LLM API latency)

---

## 12. Future Scope

- Database connectors (PostgreSQL, MySQL, BigQuery)
- Fine-tuned LLM specifically for dashboard design
- Dashboard saving and export to PDF/PowerPoint
- Cross-chart filtering (click a bar → filter all other charts)
- Multi-user with role-based access
- Scheduled data refresh and alerts

---

## 13. How to Run

```bash
# Install
pip install -r requirements.txt

# Configure
echo "HF_API_TOKEN=your_huggingface_token" > .env

# Run
streamlit run app.py
```

---

## 14. Key Numbers

| Metric | Value |
|--------|-------|
| AI dashboard generation | ~10 seconds |
| Auto dashboard generation | <1 second |
| Supported chart types | 11 |
| Max file size | 100 MB |
| Supported formats | CSV, XLSX, XLS |
| LLM | Meta Llama 3.1 8B |
| Python dependencies | 10 packages |
