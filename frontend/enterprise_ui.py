"""Enterprise tabs – Data Governance, Observability, Lineage, Schema Registry."""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.column_classifier import get_measure_columns, get_dimension_columns, classify_all_columns


def _eui_table(data: pd.DataFrame, max_rows: int = 50) -> None:
    """Render a DataFrame as a styled HTML table."""
    html = data.head(max_rows).to_html(index=False, classes="eut", border=0)
    st.markdown(f"""<div style="overflow-x:auto; max-height:420px; overflow-y:auto;
        border-radius:8px; border:1px solid #2d3148;">
        <style>
        .eut {{ width:100%; border-collapse:collapse; font-size:.82rem; font-family:'Inter',monospace; }}
        .eut th {{ background:#1e2235; color:#94a3b8; font-weight:600; padding:8px 12px; text-align:left;
                   position:sticky; top:0; border-bottom:2px solid #2d3148; font-size:.75rem;
                   text-transform:uppercase; letter-spacing:.03em; }}
        .eut td {{ padding:6px 12px; color:#e2e8f0; border-bottom:1px solid #1e2235; }}
        .eut tr:hover td {{ background:rgba(99,102,241,0.06); }}
        </style>{html}
    </div>""", unsafe_allow_html=True)

_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,29,41,0.8)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
)


# ═════════════════════════════════════════════════════════════════════════════
# DATA GOVERNANCE
# ═════════════════════════════════════════════════════════════════════════════

def _quality_score(df: pd.DataFrame) -> dict[str, Any]:
    """Compute a data quality scorecard."""
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols

    missing = df.isnull().sum().sum()
    completeness = (1 - missing / total_cells) * 100 if total_cells else 100

    duplicates = df.duplicated().sum()
    uniqueness = (1 - duplicates / n_rows) * 100 if n_rows else 100

    consistency_issues = 0
    for col in df.select_dtypes(include="object").columns:
        vals = df[col].dropna().unique()
        lower_vals = [str(v).lower().strip() for v in vals]
        if len(set(lower_vals)) < len(vals):
            consistency_issues += 1
    consistency = max(0, 100 - consistency_issues * 10)

    validity = 100.0
    for col in df.select_dtypes(include="number").columns:
        outliers = _count_outliers(df[col])
        if outliers > n_rows * 0.05:
            validity -= 5

    overall = (completeness * 0.35 + uniqueness * 0.25 + consistency * 0.2 + max(validity, 0) * 0.2)

    return {
        "overall": round(min(overall, 100), 1),
        "completeness": round(completeness, 1),
        "uniqueness": round(uniqueness, 1),
        "consistency": round(min(consistency, 100), 1),
        "validity": round(max(validity, 0), 1),
        "missing_cells": int(missing),
        "duplicate_rows": int(duplicates),
        "total_cells": total_cells,
        "consistency_issues": consistency_issues,
    }


def _count_outliers(series: pd.Series) -> int:
    clean = series.dropna()
    if len(clean) < 10:
        return 0
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((clean < lower) | (clean > upper)).sum())


def _quality_rules(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Run automated data quality rules."""
    rules = []
    n = len(df)

    for col in df.columns:
        miss_pct = df[col].isnull().mean() * 100
        if miss_pct > 0:
            status = "FAIL" if miss_pct > 20 else ("WARN" if miss_pct > 5 else "PASS")
            rules.append({
                "Rule": f"Null check: {col}",
                "Status": status,
                "Detail": f"{miss_pct:.1f}% missing ({df[col].isnull().sum():,} nulls)",
                "Severity": "High" if miss_pct > 20 else "Medium",
            })
        else:
            rules.append({
                "Rule": f"Null check: {col}",
                "Status": "PASS",
                "Detail": "No missing values",
                "Severity": "Low",
            })

    dups = df.duplicated().sum()
    rules.append({
        "Rule": "Duplicate row check",
        "Status": "FAIL" if dups > n * 0.05 else ("WARN" if dups > 0 else "PASS"),
        "Detail": f"{dups:,} duplicate rows ({dups/n*100:.1f}%)" if dups else "No duplicates",
        "Severity": "High" if dups > n * 0.05 else "Low",
    })

    for col in df.select_dtypes(include="number").columns:
        outliers = _count_outliers(df[col])
        if outliers > 0:
            rules.append({
                "Rule": f"Outlier check: {col}",
                "Status": "WARN" if outliers > n * 0.02 else "PASS",
                "Detail": f"{outliers:,} outliers detected (IQR method)",
                "Severity": "Medium" if outliers > n * 0.02 else "Low",
            })

    for col in df.select_dtypes(include="object").columns:
        vals = df[col].dropna().unique()
        lower_vals = [str(v).lower().strip() for v in vals]
        if len(set(lower_vals)) < len(vals):
            diff = len(vals) - len(set(lower_vals))
            rules.append({
                "Rule": f"Consistency check: {col}",
                "Status": "WARN",
                "Detail": f"{diff} potential case/whitespace inconsistencies",
                "Severity": "Medium",
            })

    return rules


def render_governance_tab(df: pd.DataFrame, dataset_name: str) -> None:
    """Render the Data Governance tab."""
    st.markdown("""<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:1rem;">
        <span style="font-size:1.4rem;">🛡</span>
        <div>
            <div style="font-size:1.2rem;font-weight:800;color:#f1f5f9;">Data Governance</div>
            <div style="font-size:.78rem;color:#64748b;">Quality scoring, validation rules, schema registry, and data lineage</div>
        </div>
    </div>""", unsafe_allow_html=True)

    gov1, gov2, gov3, gov4 = st.tabs(
        ["📊 Quality Score", "✅ Validation Rules", "📐 Schema Registry", "🔗 Data Lineage"]
    )

    # ── Quality Score ────────────────────────────────────────────────────
    with gov1:
        scores = _quality_score(df)

        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        _score_color = lambda s: "#22c55e" if s >= 80 else ("#f59e0b" if s >= 60 else "#ef4444")

        with sc1:
            c = _score_color(scores["overall"])
            st.markdown(f"""<div style="background:#1a1d29;border:2px solid {c};border-radius:12px;padding:1.25rem;text-align:center;">
                <div style="font-size:2rem;font-weight:800;color:{c};">{scores['overall']}%</div>
                <div style="font-size:.75rem;color:#94a3b8;font-weight:600;text-transform:uppercase;margin-top:4px;">Overall</div>
            </div>""", unsafe_allow_html=True)
        for col_w, key, label in [(sc2, "completeness", "Completeness"), (sc3, "uniqueness", "Uniqueness"),
                                   (sc4, "consistency", "Consistency"), (sc5, "validity", "Validity")]:
            with col_w:
                c = _score_color(scores[key])
                st.metric(label, f"{scores[key]}%", help=f"Quality {label.lower()}")

        st.markdown("---")

        qc1, qc2 = st.columns(2)
        with qc1:
            labels = ["Completeness", "Uniqueness", "Consistency", "Validity"]
            values = [scores["completeness"], scores["uniqueness"], scores["consistency"], scores["validity"]]
            fig = go.Figure(data=go.Scatterpolar(
                r=values + [values[0]], theta=labels + [labels[0]],
                fill="toself", fillcolor="rgba(99,102,241,0.15)",
                line=dict(color="#6366f1", width=2),
                marker=dict(size=8, color="#818cf8"),
            ))
            fig.update_layout(**_LAYOUT, title="Quality Radar", height=350,
                              polar=dict(radialaxis=dict(range=[0, 100], showticklabels=True,
                                                          tickfont=dict(size=9, color="#64748b")),
                                         angularaxis=dict(tickfont=dict(size=11, color="#f1f5f9")),
                                         bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, use_container_width=True)

        with qc2:
            col_missing = df.isnull().sum().reset_index()
            col_missing.columns = ["Column", "Missing"]
            col_missing = col_missing[col_missing["Missing"] > 0].sort_values("Missing", ascending=True)
            if len(col_missing) > 0:
                fig = px.bar(col_missing, y="Column", x="Missing", orientation="h",
                             color_discrete_sequence=["#ef4444"],
                             text=col_missing["Missing"].apply(lambda x: f"{x:,}"))
                fig.update_traces(textposition="outside", textfont_size=10)
                fig.update_layout(**_LAYOUT, title="Missing Values by Column", height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values in any column.")

        st.markdown(f"""<div style="display:flex;gap:2rem;justify-content:center;padding:1rem;
                        background:#1a1d29;border:1px solid #2d3148;border-radius:12px;">
            <div style="text-align:center;"><div style="font-size:1.3rem;font-weight:700;color:#f1f5f9;">{scores['missing_cells']:,}</div>
                <div style="font-size:.75rem;color:#64748b;">Missing Cells</div></div>
            <div style="text-align:center;"><div style="font-size:1.3rem;font-weight:700;color:#f1f5f9;">{scores['duplicate_rows']:,}</div>
                <div style="font-size:.75rem;color:#64748b;">Duplicate Rows</div></div>
            <div style="text-align:center;"><div style="font-size:1.3rem;font-weight:700;color:#f1f5f9;">{scores['total_cells']:,}</div>
                <div style="font-size:.75rem;color:#64748b;">Total Cells</div></div>
            <div style="text-align:center;"><div style="font-size:1.3rem;font-weight:700;color:#f1f5f9;">{len(df.columns)}</div>
                <div style="font-size:.75rem;color:#64748b;">Columns</div></div>
        </div>""", unsafe_allow_html=True)

    # ── Validation Rules ─────────────────────────────────────────────────
    with gov2:
        rules = _quality_rules(df)
        rules_df = pd.DataFrame(rules)

        rc1, rc2, rc3 = st.columns(3)
        pass_count = sum(1 for r in rules if r["Status"] == "PASS")
        warn_count = sum(1 for r in rules if r["Status"] == "WARN")
        fail_count = sum(1 for r in rules if r["Status"] == "FAIL")
        with rc1:
            st.metric("Rules PASS", pass_count, delta=f"{pass_count/len(rules)*100:.0f}%")
        with rc2:
            st.metric("Rules WARN", warn_count)
        with rc3:
            st.metric("Rules FAIL", fail_count)

        status_filter = st.multiselect("Filter by status", ["PASS", "WARN", "FAIL"], default=["WARN", "FAIL"], key="gov_status_filter")
        filtered_rules = rules_df[rules_df["Status"].isin(status_filter)] if status_filter else rules_df

        def _color_status(val):
            colors = {"PASS": "background-color: rgba(34,197,94,0.2); color: #22c55e",
                      "WARN": "background-color: rgba(245,158,11,0.2); color: #f59e0b",
                      "FAIL": "background-color: rgba(239,68,68,0.2); color: #ef4444"}
            return colors.get(val, "")

        _eui_table(filtered_rules)

    # ── Schema Registry ──────────────────────────────────────────────────
    with gov3:
        from schema.schema_registry import detect_schema, save_schema, load_schema, list_schema_versions

        current_schema = detect_schema(df)
        schema_df = pd.DataFrame(current_schema)

        st.markdown("**Current Schema**")
        _eui_table(schema_df)

        sc_c1, sc_c2 = st.columns([2, 1])
        with sc_c1:
            if st.button("Save Schema Snapshot", use_container_width=True, key="gov_save_schema"):
                path = save_schema(dataset_name, current_schema)
                st.success(f"Schema saved: {path.name}")

        versions = list_schema_versions(dataset_name)
        if versions:
            st.markdown("---")
            st.markdown("**Schema Version History**")
            _eui_table(pd.DataFrame(versions))

            prev = load_schema(dataset_name)
            if prev:
                prev_cols = {c["column_name"] for c in prev["schema"]}
                curr_cols = {c["column_name"] for c in current_schema}
                added = curr_cols - prev_cols
                removed = prev_cols - curr_cols
                if added or removed:
                    st.markdown("**Schema Drift Detected**")
                    if added:
                        st.warning(f"New columns: {', '.join(added)}")
                    if removed:
                        st.error(f"Removed columns: {', '.join(removed)}")
                else:
                    st.info("No schema drift — matches latest saved version.")

    # ── Data Lineage ─────────────────────────────────────────────────────
    with gov4:
        from lineage.lineage_tracker import LineageTracker, PIPELINE_STAGES

        tracker = LineageTracker(dataset_name)
        graph = tracker.get_lineage_graph_data()
        events = tracker.get_lineage()

        st.markdown("**Pipeline Flow**")

        cols = st.columns(len(PIPELINE_STAGES))
        completed_ids = {e["stage_id"] for e in events}
        for i, stage in enumerate(PIPELINE_STAGES):
            done = stage["id"] in completed_ids
            color = "#22c55e" if done else "#2d3148"
            border = f"2px solid {color}"
            with cols[i]:
                st.markdown(f"""<div style="text-align:center;padding:.75rem .25rem;border:{border};
                    border-radius:10px;background:{'rgba(34,197,94,0.08)' if done else '#1a1d29'};">
                    <div style="font-size:1.3rem;">{stage['icon']}</div>
                    <div style="font-size:.65rem;color:{'#22c55e' if done else '#64748b'};font-weight:600;margin-top:4px;">
                        {stage['label']}</div>
                </div>""", unsafe_allow_html=True)

        if events:
            st.markdown("---")
            st.markdown("**Lineage Events**")
            events_df = pd.DataFrame(events)[["timestamp", "stage_id", "stage_label"]].rename(
                columns={"timestamp": "Timestamp", "stage_id": "Stage ID", "stage_label": "Stage"})
            _eui_table(events_df)

        if st.button("Record Current Pipeline Run", use_container_width=True, key="gov_record_lineage"):
            for stage in PIPELINE_STAGES:
                if stage["id"] not in completed_ids:
                    tracker.add_event(stage["id"], {"rows": len(df), "cols": len(df.columns)})
            st.success("Pipeline lineage recorded.")
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# OBSERVABILITY
# ═════════════════════════════════════════════════════════════════════════════

def render_observability_tab(df: pd.DataFrame, profile) -> None:
    """Render the Observability tab."""
    st.markdown("""<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:1rem;">
        <span style="font-size:1.4rem;">📡</span>
        <div>
            <div style="font-size:1.2rem;font-weight:800;color:#f1f5f9;">Observability</div>
            <div style="font-size:.78rem;color:#64748b;">Data health monitoring, anomaly detection, drift analysis, and audit trail</div>
        </div>
    </div>""", unsafe_allow_html=True)

    obs1, obs2, obs3, obs4 = st.tabs(
        ["🏥 Data Health", "🔔 Anomaly Alerts", "📏 Drift Monitor", "📝 Audit Log"]
    )

    measures = get_measure_columns(df)
    dimensions = get_dimension_columns(df)

    # ── Data Health ──────────────────────────────────────────────────────
    with obs1:
        hc1, hc2, hc3, hc4 = st.columns(4)
        with hc1:
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Health Score", f"{completeness:.1f}%")
        with hc2:
            st.metric("Total Rows", f"{len(df):,}")
        with hc3:
            st.metric("Total Columns", len(df.columns))
        with hc4:
            mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
            st.metric("Memory Usage", f"{mem:.1f} MB")

        st.markdown("---")

        col_health = []
        for col in df.columns:
            s = df[col]
            missing_pct = s.isnull().mean() * 100
            unique_pct = s.nunique() / len(s) * 100 if len(s) > 0 else 0
            health = "Healthy" if missing_pct < 5 else ("Warning" if missing_pct < 20 else "Critical")
            col_health.append({
                "Column": col,
                "Type": str(s.dtype),
                "Missing %": round(missing_pct, 1),
                "Unique %": round(unique_pct, 1),
                "Health": health,
            })

        health_df = pd.DataFrame(col_health)
        st.markdown("**Column Health Report**")
        _eui_table(health_df)

        if measures:
            st.markdown("---")
            st.markdown("**Distribution Health**")
            dc1, dc2 = st.columns(2)
            with dc1:
                skew_data = []
                for m in measures[:10]:
                    sk = df[m].skew()
                    skew_data.append({"Measure": m, "Skewness": round(sk, 2),
                                      "Status": "Normal" if abs(sk) < 1 else ("Skewed" if abs(sk) < 2 else "Highly Skewed")})
                _eui_table(pd.DataFrame(skew_data))

            with dc2:
                if len(measures) >= 2:
                    stats_data = []
                    for m in measures[:10]:
                        cv = df[m].std() / df[m].mean() * 100 if df[m].mean() != 0 else 0
                        stats_data.append({"Measure": m, "CV %": round(cv, 1), "Std Dev": round(df[m].std(), 2),
                                           "Stability": "Stable" if cv < 50 else ("Variable" if cv < 100 else "Volatile")})
                    _eui_table(pd.DataFrame(stats_data))

    # ── Anomaly Alerts ───────────────────────────────────────────────────
    with obs2:
        alerts = []
        for col in measures:
            s = df[col].dropna()
            if len(s) < 10:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            outlier_count = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
            if outlier_count > 0:
                pct = outlier_count / len(s) * 100
                severity = "Critical" if pct > 5 else ("Warning" if pct > 1 else "Info")
                alerts.append({
                    "Column": col,
                    "Anomalies": int(outlier_count),
                    "Percentage": f"{pct:.1f}%",
                    "Severity": severity,
                    "Range": f"[{q1 - 1.5*iqr:,.2f}, {q3 + 1.5*iqr:,.2f}]",
                    "Min": f"{s.min():,.2f}",
                    "Max": f"{s.max():,.2f}",
                })

        if alerts:
            ac1, ac2, ac3 = st.columns(3)
            crit = sum(1 for a in alerts if a["Severity"] == "Critical")
            warn = sum(1 for a in alerts if a["Severity"] == "Warning")
            info = sum(1 for a in alerts if a["Severity"] == "Info")
            with ac1:
                st.metric("Critical Alerts", crit)
            with ac2:
                st.metric("Warning Alerts", warn)
            with ac3:
                st.metric("Info Alerts", info)

            _eui_table(pd.DataFrame(alerts))

            st.markdown("---")
            selected_alert = st.selectbox("Inspect column", [a["Column"] for a in alerts], key="obs_alert_col")
            if selected_alert:
                s = df[selected_alert].dropna()
                fig = px.histogram(s, nbins=50, color_discrete_sequence=["#6366f1"])
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                fig.add_vline(x=q1 - 1.5 * iqr, line_dash="dash", line_color="#ef4444", annotation_text="Lower bound")
                fig.add_vline(x=q3 + 1.5 * iqr, line_dash="dash", line_color="#ef4444", annotation_text="Upper bound")
                fig.update_layout(**_LAYOUT, title=f"{selected_alert} Distribution with Anomaly Bounds", height=350)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No anomalies detected across all numeric columns.")

    # ── Drift Monitor ────────────────────────────────────────────────────
    with obs3:
        st.markdown("**Statistical Drift Analysis**")
        st.caption("Compares first half vs second half of the dataset to detect distribution shifts")

        if len(df) >= 20:
            mid = len(df) // 2
            first_half = df.iloc[:mid]
            second_half = df.iloc[mid:]

            drift_results = []
            for col in measures[:15]:
                mean1, mean2 = first_half[col].mean(), second_half[col].mean()
                std1, std2 = first_half[col].std(), second_half[col].std()
                change_pct = ((mean2 - mean1) / abs(mean1) * 100) if mean1 != 0 else 0
                drift = "Significant" if abs(change_pct) > 20 else ("Moderate" if abs(change_pct) > 10 else "Stable")
                drift_results.append({
                    "Column": col,
                    "Mean (1st half)": round(mean1, 2),
                    "Mean (2nd half)": round(mean2, 2),
                    "Change %": round(change_pct, 1),
                    "Drift": drift,
                })

            drift_df = pd.DataFrame(drift_results)
            _eui_table(drift_df)

            sig_drift = drift_df[drift_df["Drift"] != "Stable"]
            if len(sig_drift) > 0:
                fig = px.bar(sig_drift, x="Column", y="Change %", color="Drift",
                             color_discrete_map={"Significant": "#ef4444", "Moderate": "#f59e0b", "Stable": "#22c55e"})
                fig.update_layout(**_LAYOUT, title="Distribution Drift by Column", height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No significant drift detected.")

            if dimensions:
                st.markdown("---")
                st.markdown("**Categorical Drift**")
                cat_drift = []
                for col in dimensions[:10]:
                    dist1 = first_half[col].value_counts(normalize=True)
                    dist2 = second_half[col].value_counts(normalize=True)
                    all_vals = set(dist1.index) | set(dist2.index)
                    max_shift = 0
                    for val in all_vals:
                        p1 = dist1.get(val, 0)
                        p2 = dist2.get(val, 0)
                        max_shift = max(max_shift, abs(p2 - p1) * 100)
                    cat_drift.append({
                        "Column": col,
                        "Categories": len(all_vals),
                        "Max Shift %": round(max_shift, 1),
                        "Status": "Drifted" if max_shift > 10 else "Stable",
                    })
                _eui_table(pd.DataFrame(cat_drift))
        else:
            st.info("Dataset too small for drift analysis (need 20+ rows).")

    # ── Audit Log ────────────────────────────────────────────────────────
    with obs4:
        now = dt.datetime.now()
        audit_entries = [
            {"Timestamp": now.strftime("%Y-%m-%d %H:%M:%S"), "Action": "Dataset loaded",
             "Details": f"{len(df):,} rows x {len(df.columns)} cols", "User": "system"},
            {"Timestamp": now.strftime("%Y-%m-%d %H:%M:%S"), "Action": "Schema detected",
             "Details": f"{len(df.columns)} columns classified", "User": "system"},
            {"Timestamp": now.strftime("%Y-%m-%d %H:%M:%S"), "Action": "Quality scan",
             "Details": f"Score: {_quality_score(df)['overall']}%", "User": "system"},
            {"Timestamp": now.strftime("%Y-%m-%d %H:%M:%S"), "Action": "Profiling complete",
             "Details": f"{len(measures)} measures, {len(dimensions)} dimensions", "User": "system"},
        ]

        dt_cols = df.select_dtypes(include="datetime").columns.tolist()
        if dt_cols:
            audit_entries.append({
                "Timestamp": now.strftime("%Y-%m-%d %H:%M:%S"), "Action": "Time series detected",
                "Details": f"Column: {dt_cols[0]}", "User": "system",
            })

        for col in measures[:3]:
            outlier_n = _count_outliers(df[col])
            if outlier_n > 0:
                audit_entries.append({
                    "Timestamp": now.strftime("%Y-%m-%d %H:%M:%S"), "Action": "Anomaly detected",
                    "Details": f"{outlier_n} outliers in {col}", "User": "auto-monitor",
                })

        _eui_table(pd.DataFrame(audit_entries))

        st.markdown("---")
        st.markdown(f"""<div style="display:flex;gap:2rem;padding:1rem;
                        background:#1a1d29;border:1px solid #2d3148;border-radius:12px;">
            <div style="text-align:center;flex:1;"><div style="font-size:1.5rem;font-weight:700;color:#22c55e;">{len(audit_entries)}</div>
                <div style="font-size:.75rem;color:#64748b;">Events Logged</div></div>
            <div style="text-align:center;flex:1;"><div style="font-size:1.5rem;font-weight:700;color:#6366f1;">Auto</div>
                <div style="font-size:.75rem;color:#64748b;">Monitoring Mode</div></div>
            <div style="text-align:center;flex:1;"><div style="font-size:1.5rem;font-weight:700;color:#f1f5f9;">Real-time</div>
                <div style="font-size:.75rem;color:#64748b;">Alert Frequency</div></div>
        </div>""", unsafe_allow_html=True)
