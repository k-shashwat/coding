"""
Parses bank statements from CSV, Excel, or PDF into a normalized DataFrame.
Output always has columns: date, description, amount (negative = debit, positive = credit)
"""

import re
import io
import pandas as pd
import pdfplumber


# ── Column name synonyms banks commonly use ──────────────────────────────────
_DATE_ALIASES = {"date", "transaction date", "txn date", "value date", "posted date"}
_DESC_ALIASES = {"description", "narration", "details", "particulars", "remarks",
                 "transaction description", "merchant", "payee", "memo"}
_AMOUNT_ALIASES = {"amount", "transaction amount", "txn amount"}
_DEBIT_ALIASES = {"debit", "debit amount", "withdrawal", "dr"}
_CREDIT_ALIASES = {"credit", "credit amount", "deposit", "cr"}


def _normalize_col(name: str) -> str:
    return name.strip().lower()


def _find_col(cols: list[str], aliases: set) -> str | None:
    for c in cols:
        if _normalize_col(c) in aliases:
            return c
    return None


def _clean_amount(val) -> float:
    """Remove currency symbols, commas, parentheses (negatives), then cast."""
    if pd.isna(val):
        return 0.0
    s = str(val).strip().replace(",", "").replace(" ", "")
    s = s.replace("$", "").replace("€", "").replace("£", "").replace("₹", "")
    # (123.45) → -123.45
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return 0.0


def _build_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Map whatever columns exist → date / description / amount."""
    cols = list(df.columns)
    norm = {c: _normalize_col(c) for c in cols}

    date_col = _find_col(cols, _DATE_ALIASES)
    desc_col = _find_col(cols, _DESC_ALIASES)

    # Try a single amount column first
    amount_col = _find_col(cols, _AMOUNT_ALIASES)
    debit_col = _find_col(cols, _DEBIT_ALIASES)
    credit_col = _find_col(cols, _CREDIT_ALIASES)

    if date_col is None or desc_col is None:
        raise ValueError(
            f"Could not identify Date / Description columns.\n"
            f"Found columns: {cols}\n"
            "Please rename your columns to include 'Date' and 'Description'."
        )

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], infer_datetime_format=True, errors="coerce")
    out["description"] = df[desc_col].astype(str).str.strip()

    if amount_col:
        out["amount"] = df[amount_col].apply(_clean_amount)
    elif debit_col and credit_col:
        debits = df[debit_col].apply(_clean_amount)
        credits = df[credit_col].apply(_clean_amount)
        # Convention: debits are negative, credits positive
        out["amount"] = credits - debits
    else:
        raise ValueError(
            "Could not find an Amount, or Debit/Credit column pair.\n"
            f"Found columns: {cols}"
        )

    out = out.dropna(subset=["date"]).reset_index(drop=True)
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def parse_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return _build_normalized(df)


def parse_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    return _build_normalized(df)


def parse_pdf(file) -> pd.DataFrame:
    """
    Extracts tables from a PDF bank statement.
    Tries every page; concatenates all table rows found.
    """
    rows = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue
                # First row is likely the header
                header = [str(h).strip() if h else f"col_{i}"
                          for i, h in enumerate(table[0])]
                for row in table[1:]:
                    if row:
                        rows.append(dict(zip(header, row)))

    if not rows:
        raise ValueError(
            "No tables found in the PDF. "
            "Make sure the PDF contains machine-readable text (not a scanned image)."
        )

    df = pd.DataFrame(rows)
    return _build_normalized(df)


def parse_file(uploaded_file) -> pd.DataFrame:
    """Dispatch to the right parser based on file extension."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return parse_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return parse_excel(uploaded_file)
    elif name.endswith(".pdf"):
        return parse_pdf(io.BytesIO(uploaded_file.read()))
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")
