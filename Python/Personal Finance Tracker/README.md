# 💰 Personal Finance Tracker

Upload any bank statement (CSV, Excel, or PDF) and instantly get:
- **Auto-categorization** using semantic embeddings (no API key needed)
- **Spending charts** — pie, bar, and monthly trends
- **Anomaly detection** — flags unusual spends and duplicate transactions
- **Downloadable** categorized CSV

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Embeddings | `intfloat/e5-small-v2` (HuggingFace, runs locally) |
| Anomaly Detection | `IsolationForest` (scikit-learn) |
| Charts | Plotly |
| File parsing | pandas, pdfplumber, openpyxl |

---

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd "Personal Finance Tracker"

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The E5 model (~90 MB) is downloaded automatically on first run and cached locally.

---

## Supported File Formats

| Format | Requirements |
|---|---|
| **CSV** | Needs `Date`, `Description`, and `Amount` (or `Debit`/`Credit`) columns |
| **Excel** (.xlsx/.xls) | Same column requirements as CSV |
| **PDF** | Must be machine-readable (not a scanned image); app extracts tables |

A sample statement is included: `sample_statement.csv`

---

## Categories

The app auto-classifies transactions into 13 categories:

Food & Dining · Transport · Shopping · Entertainment · Healthcare ·  
Utilities & Bills · Housing · Education · Travel · Finance & Insurance ·  
Income & Refund · Personal Care · Transfers · Other

---

## How Categorization Works

1. Transaction descriptions are embedded as **query vectors** using E5-small
2. Each category has 4–6 representative example phrases embedded as **passage vectors**
3. The transaction is assigned the category with the **highest cosine similarity**
4. No API key, no internet required after the first model download

---

## Anomaly Detection

Two methods run in parallel:
- **IsolationForest** per category — flags transactions that are statistical outliers within their category
- **Duplicate detector** — catches identical transactions (same date + description + amount)
