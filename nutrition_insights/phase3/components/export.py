# phase3/components/export.py
from __future__ import annotations

import io
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from utils.common import pretty_int


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")


def _filter_by_source(df: pd.DataFrame, source_filter: str) -> pd.DataFrame:
    if source_filter == "All":
        return df
    s = str(source_filter).lower()
    return df[df["source"].astype(str).str.lower() == s].copy()


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Keep the essentials if present
    cols = [c for c in ["date", "source", "title", "url", "summary", "text"] if c in df.columns]
    return df[cols].copy() if cols else df.copy()


def _make_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")




def render(df: pd.DataFrame, source_filter: str, window_days: int) -> None:
    st.subheader("Download / Export")

    fdf = _filter_by_source(df, source_filter)
    fdf = _select_columns(fdf)
    n = len(fdf)

    st.caption(
        f"Preparing export for **{source_filter}** over last **{window_days}** day(s). "
        f"Rows: **{pretty_int(n)}**"
    )

    # CSV
    csv_bytes = _make_csv(fdf)
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_bytes,
        file_name=f"proteinscope_{source_filter.lower()}_{_utc_stamp()}.csv",
        mime="text/csv",
        use_container_width=True,
    )
