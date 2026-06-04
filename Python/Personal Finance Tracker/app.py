"""
Personal Finance Tracker
========================
Upload a bank statement (CSV / Excel / PDF) → auto-categorize → detect anomalies → explore.
"""

import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from parser import parse_file
from categorizer import categorize
from anomaly import detect_anomalies, anomaly_summary


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Personal Finance Tracker",
    page_icon="💰",
    layout="wide",
)

st.title("💰 Personal Finance Tracker")
st.caption("Upload your bank statement → get instant categorization, insights, and anomaly detection.")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Statement")
    uploaded = st.file_uploader(
        "Supported formats: CSV, Excel (.xlsx/.xls), PDF",
        type=["csv", "xlsx", "xls", "pdf"],
    )

    st.divider()
    st.markdown("""
**How it works**
1. Your file is parsed into a standard format
2. Transaction descriptions are embedded using **E5-small** (HuggingFace)
3. Each transaction is matched to the closest category by cosine similarity
4. **IsolationForest** flags statistically unusual spends per category
""")

    st.divider()
    st.markdown("""
**Expected columns** (any common bank format works):
- `Date` or `Transaction Date`
- `Description` or `Narration` or `Particulars`
- `Amount` **or** separate `Debit` / `Credit` columns
""")


# ── Main flow ─────────────────────────────────────────────────────────────────
if uploaded is None:
    st.info("👈 Upload a bank statement from the sidebar to get started.")

    # Show a sample of what the app expects
    st.subheader("Sample CSV format")
    sample = pd.DataFrame({
        "Date": ["2024-01-05", "2024-01-06", "2024-01-07"],
        "Description": ["SWIGGY ORDER 9823", "UBER TRIP BLR", "HDFC SALARY CREDIT"],
        "Debit": [450, 220, 0],
        "Credit": [0, 0, 85000],
    })
    st.dataframe(sample, use_container_width=True)
    st.stop()


# ── Parse ─────────────────────────────────────────────────────────────────────
with st.spinner("Parsing your statement…"):
    try:
        df = parse_file(uploaded)
    except ValueError as e:
        st.error(f"**Parse error:** {e}")
        st.stop()

st.success(f"Parsed **{len(df):,} transactions** from `{uploaded.name}`")


# ── Categorize ────────────────────────────────────────────────────────────────
with st.spinner("Categorizing transactions with E5 embeddings…"):
    df["category"] = categorize(df["description"].tolist())


# ── Anomaly detection ─────────────────────────────────────────────────────────
with st.spinner("Running anomaly detection…"):
    df = detect_anomalies(df)


# ── Date range filter ─────────────────────────────────────────────────────────
st.subheader("Filters")
col1, col2 = st.columns(2)
min_date, max_date = df["date"].min().date(), df["date"].max().date()
start = col1.date_input("From", value=min_date, min_value=min_date, max_value=max_date)
end = col2.date_input("To", value=max_date, min_value=min_date, max_value=max_date)

mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
fdf = df[mask].copy()

if fdf.empty:
    st.warning("No transactions in the selected date range.")
    st.stop()


# ── KPI cards ─────────────────────────────────────────────────────────────────
st.divider()
debits = fdf[fdf["amount"] < 0]["amount"].sum()
credits = fdf[fdf["amount"] > 0]["amount"].sum()
net = fdf["amount"].sum()
anomaly_count = fdf["is_anomaly"].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Spent", f"₹{abs(debits):,.0f}")
k2.metric("Total Received", f"₹{credits:,.0f}")
k3.metric("Net Balance Change", f"₹{net:,.0f}", delta_color="normal")
k4.metric("Anomalies Flagged", int(anomaly_count),
          delta="⚠️ review needed" if anomaly_count else "✅ all clear",
          delta_color="inverse" if anomaly_count else "off")


# ── Tabs ──────────────────────────────────────────────────────────────────────
st.divider()
tab1, tab2, tab3, tab4 = st.tabs(["📊 Spending Breakdown", "📅 Timeline", "⚠️ Anomalies", "📋 Full Ledger"])


# ── Tab 1: Spending breakdown ─────────────────────────────────────────────────
with tab1:
    spend_df = fdf[fdf["amount"] < 0].copy()
    spend_df["abs_amount"] = spend_df["amount"].abs()

    by_cat = spend_df.groupby("category")["abs_amount"].sum().reset_index()
    by_cat.columns = ["Category", "Amount"]
    by_cat = by_cat.sort_values("Amount", ascending=False)

    c1, c2 = st.columns([1, 1])

    with c1:
        fig_pie = px.pie(
            by_cat, values="Amount", names="Category",
            title="Spend by Category",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        fig_bar = px.bar(
            by_cat, x="Amount", y="Category", orientation="h",
            title="Amount Spent per Category",
            color="Amount",
            color_continuous_scale="Blues",
        )
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"},
                              coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Category Summary Table")
    total_spend = by_cat["Amount"].sum()
    by_cat["% of Total"] = (by_cat["Amount"] / total_spend * 100).round(1).astype(str) + "%"
    by_cat["Amount"] = by_cat["Amount"].apply(lambda x: f"₹{x:,.0f}")
    st.dataframe(by_cat, use_container_width=True, hide_index=True)


# ── Tab 2: Timeline ───────────────────────────────────────────────────────────
with tab2:
    timeline_df = fdf.copy()
    timeline_df["month"] = timeline_df["date"].dt.to_period("M").astype(str)

    monthly = (
        timeline_df[timeline_df["amount"] < 0]
        .groupby(["month", "category"])["amount"]
        .sum()
        .abs()
        .reset_index()
    )
    monthly.columns = ["Month", "Category", "Amount"]

    fig_timeline = px.bar(
        monthly, x="Month", y="Amount", color="Category",
        title="Monthly Spending by Category",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Running balance line chart
    fdf_sorted = fdf.sort_values("date")
    fdf_sorted["running_balance"] = fdf_sorted["amount"].cumsum()
    fig_balance = px.line(
        fdf_sorted, x="date", y="running_balance",
        title="Running Balance Change Over Time",
        labels={"running_balance": "Cumulative Amount (₹)", "date": "Date"},
    )
    fig_balance.update_traces(line_color="#2196F3")
    st.plotly_chart(fig_balance, use_container_width=True)


# ── Tab 3: Anomalies ──────────────────────────────────────────────────────────
with tab3:
    anom = anomaly_summary(fdf)

    if anom.empty:
        st.success("No anomalies detected in this date range.")
    else:
        st.warning(f"**{len(anom)} suspicious transactions** found.")

        # Highlight anomalies in a styled table
        def color_amount(val):
            return "color: red" if val < 0 else "color: green"

        styled = anom.copy()
        styled["date"] = styled["date"].dt.strftime("%Y-%m-%d")
        styled["amount"] = styled["amount"].apply(lambda x: f"₹{x:,.0f}")
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.subheader("Anomalies by Category")
        anom_counts = anom.groupby("category").size().reset_index(name="count")
        fig_anom = px.bar(anom_counts, x="category", y="count",
                          color="count", color_continuous_scale="Reds",
                          title="Anomaly Count per Category")
        fig_anom.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_anom, use_container_width=True)


# ── Tab 4: Full ledger ────────────────────────────────────────────────────────
with tab4:
    st.subheader("All Transactions")

    # Category filter
    all_cats = ["All"] + sorted(fdf["category"].unique().tolist())
    selected_cat = st.selectbox("Filter by category", all_cats)

    view_df = fdf if selected_cat == "All" else fdf[fdf["category"] == selected_cat]
    view_df = view_df.sort_values("date", ascending=False).copy()

    # Format for display
    display = view_df[["date", "description", "category", "amount", "is_anomaly", "anomaly_reason"]].copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    display["amount"] = display["amount"].apply(lambda x: f"₹{x:,.0f}")
    display = display.rename(columns={
        "is_anomaly": "⚠️",
        "anomaly_reason": "Anomaly Note",
    })

    st.dataframe(display, use_container_width=True, hide_index=True)

    # Download button
    csv_bytes = view_df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download categorized CSV",
        data=csv_bytes,
        file_name="categorized_transactions.csv",
        mime="text/csv",
    )
