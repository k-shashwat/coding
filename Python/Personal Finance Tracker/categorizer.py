"""
Semantic transaction categorizer using intfloat/e5-small-v2.

E5 models expect inputs prefixed with "query: " or "passage: ".
We embed category examples as passages and transaction descriptions as queries,
then pick the closest category by cosine similarity.
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

MODEL_NAME = "intfloat/e5-small-v2"

# ── Category definitions ──────────────────────────────────────────────────────
# Each category has several representative phrases so the embedding space is
# well-anchored even for unusual merchant names.
CATEGORIES: dict[str, list[str]] = {
    "Food & Dining": [
        "restaurant meal food dining",
        "swiggy zomato food delivery",
        "grocery supermarket vegetables fruits",
        "cafe coffee bakery",
        "hotel buffet lunch dinner",
        "dominos pizza subway burger",
    ],
    "Transport": [
        "uber ola taxi cab ride",
        "petrol diesel fuel gas station",
        "metro bus train ticket fare",
        "toll parking auto rickshaw",
        "flight airline airways boarding",
        "rapido bike taxi",
    ],
    "Shopping": [
        "amazon flipkart myntra online shopping",
        "retail store clothing apparel fashion",
        "electronics gadget mobile laptop",
        "zepto blinkit quick commerce",
        "department store mall purchase",
    ],
    "Entertainment": [
        "netflix amazon prime hotstar streaming",
        "spotify apple music youtube premium",
        "movie cinema theatre ticket",
        "gaming steam playstation xbox",
        "concert event show ticket booking",
    ],
    "Healthcare": [
        "pharmacy medicine chemist drugstore",
        "hospital clinic doctor consultation",
        "lab test diagnostic pathology",
        "health insurance medical premium",
        "dentist eye care optical",
    ],
    "Utilities & Bills": [
        "electricity bill power supply board",
        "water bill municipal corporation",
        "internet broadband wifi airtel jio",
        "mobile recharge prepaid postpaid",
        "gas cylinder LPG pipeline",
        "DTH cable TV subscription",
    ],
    "Housing": [
        "rent apartment flat house lease",
        "home loan mortgage EMI property",
        "maintenance society charges apartment",
        "furniture home decor ikea",
        "plumber electrician repair service",
    ],
    "Education": [
        "school fees tuition college university",
        "coursera udemy online course learning",
        "books stationery library subscription",
        "exam fee certification training",
    ],
    "Travel": [
        "hotel booking airbnb oyo stay",
        "makemytrip goibibo flight hotel booking",
        "travel agency tour package holiday",
        "passport visa travel insurance",
        "foreign currency forex exchange",
    ],
    "Finance & Insurance": [
        "insurance premium LIC policy",
        "SIP mutual fund investment zerodha",
        "bank charges fees service tax",
        "credit card payment due",
        "loan repayment interest principal",
        "fixed deposit FD RD savings",
    ],
    "Income & Refund": [
        "salary credit payroll employer",
        "freelance payment client invoice",
        "refund cashback reversal return",
        "interest credit bank FD maturity",
        "dividend bonus incentive",
    ],
    "Personal Care": [
        "salon haircut spa beauty parlour",
        "gym fitness yoga membership",
        "nykaa cosmetics personal care products",
    ],
    "Transfers": [
        "NEFT IMPS RTGS wire transfer",
        "UPI transfer send money",
        "self transfer between accounts",
        "wallet paytm phonepe gpay load",
    ],
    "Other": [
        "miscellaneous unknown unclassified",
    ],
}


@st.cache_resource(show_spinner="Loading E5 model (one-time download ~90 MB)…")
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@st.cache_data(show_spinner="Building category embeddings…")
def _build_category_embeddings(_model: SentenceTransformer) -> tuple[list[str], np.ndarray]:
    """
    For each category, embed all its example phrases and average them into
    a single representative vector.
    """
    cat_names = list(CATEGORIES.keys())
    cat_vectors = []
    for phrases in CATEGORIES.values():
        passages = [f"passage: {p}" for p in phrases]
        vecs = _model.encode(passages, normalize_embeddings=True, show_progress_bar=False)
        cat_vectors.append(vecs.mean(axis=0))
    cat_matrix = np.vstack(cat_vectors)  # shape: (n_categories, embedding_dim)
    return cat_names, cat_matrix


def categorize(descriptions: list[str]) -> list[str]:
    """Return a category label for each transaction description."""
    model = _load_model()
    cat_names, cat_matrix = _build_category_embeddings(model)

    queries = [f"query: {d}" for d in descriptions]
    query_vecs = model.encode(queries, normalize_embeddings=True,
                              show_progress_bar=False, batch_size=64)

    sims = cosine_similarity(query_vecs, cat_matrix)  # (n_txns, n_categories)
    best_idx = sims.argmax(axis=1)
    return [cat_names[i] for i in best_idx]
