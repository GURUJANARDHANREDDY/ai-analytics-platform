"""Chat-with-data UI component (modern SaaS style with suggestion chips)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from backend.ai_chat_engine import ask_question


def _html_table(data: pd.DataFrame, max_rows: int = 50) -> None:
    """Render a DataFrame as a styled HTML table (visible on dark theme)."""
    html = data.head(max_rows).to_html(index=False, classes="ht", border=0)
    st.markdown(f"""<div style="overflow-x:auto; max-height:400px; overflow-y:auto;
        border-radius:8px; border:1px solid #2d3148;">
        <style>
        .ht {{ width:100%; border-collapse:collapse; font-size:.82rem; font-family:'Inter',monospace; }}
        .ht th {{ background:#1e2235; color:#94a3b8; font-weight:600; padding:8px 12px; text-align:left;
                  position:sticky; top:0; border-bottom:2px solid #2d3148; font-size:.75rem;
                  text-transform:uppercase; letter-spacing:.03em; }}
        .ht td {{ padding:6px 12px; color:#e2e8f0; border-bottom:1px solid #1e2235; }}
        .ht tr:hover td {{ background:rgba(99,102,241,0.06); }}
        </style>{html}
    </div>""", unsafe_allow_html=True)


_SUGGESTIONS = [
    "How many rows are in the dataset?",
    "What is the average of the first numeric column?",
    "Show a summary of the dataset",
    "Which column has the most missing values?",
    "What is the maximum value?",
]


def render_chat_interface(df: pd.DataFrame) -> None:
    """Render the conversational data Q&A interface with modern styling."""

    st.markdown("""<div class="section-header">
        <div class="section-icon icon-blue">💬</div>
        <h2>Chat with Your Data</h2>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.85rem; color:#64748b; margin-bottom:1rem; line-height:1.6;">
        Ask questions about your dataset in plain English. The AI will analyze your data and respond with answers, tables, or visualizations.
    </div>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Suggestion chips when chat is empty
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="font-size:0.75rem; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.6rem;">
            Try asking
        </div>
        """, unsafe_allow_html=True)

        chip_cols = st.columns(len(_SUGGESTIONS))
        for i, (col, suggestion) in enumerate(zip(chip_cols, _SUGGESTIONS)):
            with col:
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    _handle_question(df, suggestion)
                    st.rerun()

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user", avatar="👤"):
            st.markdown(entry["question"])
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(entry["answer"])
            if entry.get("data") is not None:
                _html_table(entry["data"])

    # Input
    question = st.chat_input("Ask a question about your dataset…")

    if question:
        _handle_question(df, question)
        st.rerun()


def _handle_question(df: pd.DataFrame, question: str) -> None:
    """Process a question and append to chat history."""
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.status("Analyzing your data…", expanded=True) as status:
            st.write("Interpreting question…")
            response = ask_question(df, question)
            status.update(label="Done!", state="complete", expanded=False)

        st.markdown(response.answer)
        if response.data is not None:
            _html_table(response.data.head(50))

    st.session_state.chat_history.append({
        "question": question,
        "answer": response.answer,
        "data": response.data.head(50) if response.data is not None else None,
    })
