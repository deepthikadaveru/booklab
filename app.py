# app.py
# Professional Streamlit Book Predictor & Recommender (image-rich, polished)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
from io import BytesIO
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from functools import lru_cache

# -------------------------
# Config / Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "books_cleaned_featured.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REG_MODEL_PATH = os.path.join(MODEL_DIR, "rf_regressor.pkl")
CLF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_classifier.pkl")
SIM_MATRIX_PATH = os.path.join(MODEL_DIR, "similarity_matrix.pkl")

st.set_page_config(page_title="📚 BookLab: Predictor & Recommender", layout="wide",
                   initial_sidebar_state="expanded")

# -------------------------
# Utilities: cover fetching
# -------------------------
# We will try Google Books API first (no key required for simple queries),
# then fall back to Open Library.
HEADERS = {"User-Agent": "BookLab-App/1.0 (+https://yourproject.example)"}

@st.cache_data(show_spinner=False, ttl=60*60*24)
def fetch_cover_google(title, author=None):
    """Return thumbnail url via Google Books API or None."""
    q = title
    if author:
        q += f" {author}"
    q = q.replace(" ", "+")
    url = f"https://www.googleapis.com/books/v1/volumes?q={q}&maxResults=3"
    try:
        r = requests.get(url, headers=HEADERS, timeout=6)
        r.raise_for_status()
        data = r.json()
        items = data.get("items")
        if items:
            for item in items:
                v = item.get("volumeInfo", {})
                # prefer thumbnail or smallThumbnail
                links = v.get("imageLinks", {})
                if links.get("thumbnail"):
                    return links.get("thumbnail").replace("http://", "https://")
                if links.get("smallThumbnail"):
                    return links.get("smallThumbnail").replace("http://", "https://")
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False, ttl=60*60*24)
def fetch_cover_openlibrary(title):
    """Return OpenLibrary cover URL (may 404 if not found)."""
    # OpenLibrary expects identifiers; try a simple pattern - may not always work
    t = title.replace(" ", "+")
    url = f"https://covers.openlibrary.org/b/title/{t}-L.jpg"
    # quick check
    try:
        r = requests.get(url, headers=HEADERS, timeout=5)
        if r.status_code == 200:
            return url
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False, ttl=60*60*24)
def fetch_cover_image(title, author=None):
    """Try Google Books first, then OpenLibrary. Return PIL.Image or None."""
    url = fetch_cover_google(title, author)
    if not url:
        url = fetch_cover_openlibrary(title)
    if not url:
        return None
    try:
        r = requests.get(url, headers=HEADERS, timeout=6)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

# -------------------------
# Load dataset & models
# -------------------------
@st.cache_resource
def load_resources():
    # Data
    df_local = pd.read_csv(DATA_PATH)
    # Ensure consistent columns
    required_cols = ['title','authors','average_rating','ratings_count','text_reviews_count',
                     'author_influence','publisher_influence','book_age','publisher','language_code','num_pages']
    for c in required_cols:
        if c not in df_local.columns:
            df_local[c] = np.nan
    # Models & sim matrix
    if not os.path.exists(REG_MODEL_PATH) or not os.path.exists(CLF_MODEL_PATH) or not os.path.exists(SIM_MATRIX_PATH):
        raise FileNotFoundError("Model files or sim matrix not found in models/ (rf_regressor.pkl, rf_classifier.pkl, similarity_matrix.pkl)")
    reg = joblib.load(REG_MODEL_PATH)
    clf = joblib.load(CLF_MODEL_PATH)
    sim = joblib.load(SIM_MATRIX_PATH)
    return df_local, reg, clf, sim

try:
    df, rf_reg, rf_clf, similarity_matrix = load_resources()
except Exception as e:
    st.error("Error loading resources: " + str(e))
    st.stop()

# Normalize title index for faster search
title_to_idx = {t: i for i,t in enumerate(df['title'].astype(str).values)}

# Feature list (must match features used for training)
FEATURES = ['ratings_count','text_reviews_count','author_influence','book_age','publisher_influence']

# -------------------------
# Helper functions
# -------------------------
@st.cache_data(show_spinner=False)
def recommend_titles_by_index(idx, top_n=6):
    sims = list(enumerate(similarity_matrix[idx]))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recs = []
    for i,score in sims:
        recs.append({
            "title": df.iloc[i]['title'],
            "authors": df.iloc[i]['authors'],
            "score": float(score),
            "rating": df.iloc[i]['average_rating'],
            "ratings_count": int(df.iloc[i]['ratings_count'])
        })
    return recs

@st.cache_data(show_spinner=False)
def get_feature_importances(_model, features):
    try:
        importances = _model.feature_importances_
        return pd.DataFrame({
            "feature": features,
            "importance": importances
        }).sort_values("importance", ascending=False)
    except Exception:
        return pd.DataFrame({
            "feature": features,
            "importance": [0] * len(features)
        })

# -------------------------
# Layout: Header
# -------------------------
st.markdown("""<style>
.header {background: linear-gradient(90deg,#4b6cb7,#182848); padding:20px; border-radius:10px; color:white}
.bookcard {border-radius:8px; box-shadow: 0 3px 8px rgba(0,0,0,0.15); padding:10px; background: #fff;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>📚 BookLab — Popularity Predictor & Recommender</h1><p style="font-size:14px">Predict ratings, find similar books, explore top authors & publishers. Built with ML & live covers.</p></div>', unsafe_allow_html=True)
st.write("")  # spacing

# -------------------------
# Main: Input & Action
# -------------------------
col_main, col_right = st.columns([2.2,1])

with col_main:
    st.subheader("Find a Book")
    book_input = st.text_input("Type a book title (autocomplete not implemented):", placeholder="e.g. The Hobbit")
    st.caption("Tip: use exact title from dataset for best matches. Partial matches will attempt to find close title.")
    submit = st.button("Predict & Recommend", type="primary")

with col_right:
    st.subheader("Quick Filters")
    lang_filter = st.multiselect("Language", options=sorted(df['language_code'].dropna().unique()), default=[])
    pub_filter = st.selectbox("Publisher (top)", options=["All"] + list(df['publisher'].value_counts().head(20).index))
    # Handle missing page counts safely
min_pages_val = int(df['num_pages'].fillna(0).min())
max_pages_val = int(df['num_pages'].fillna(0).max() if not df['num_pages'].isna().all() else 1000)

min_pages, max_pages = st.slider(
    "Pages range",
    min_pages_val,
    max_pages_val if max_pages_val > 0 else 1000,
    (0, min(1000, max_pages_val))
)

    # Apply filters when exploring table below

# -------------------------
# Perform prediction & show results
# -------------------------
if submit:
    # try exact match first, otherwise best fuzzy
    key = book_input.strip()
    idx = title_to_idx.get(key)
    if idx is None:
        # try a case-insensitive match or substring match
        matches = df[df['title'].str.contains(key, case=False, na=False)]
        if len(matches) > 0:
            idx = matches.index[0]
        else:
            # fallback: try startswith or best levenshtein-like (simple)
            candidates = df['title'].tolist()
            idx = None
    if idx is None:
        st.warning("Book not found in dataset. Try a different title or check spelling. Showing top-rated books instead.")
    else:
        # prepare feature row
        X_row = df.loc[[idx], FEATURES].fillna(0)
        # predictions
        try:
            pred_rating = float(rf_reg.predict(X_row)[0])
        except Exception:
            pred_rating = float(df.iloc[idx]['average_rating'])
        try:
            pred_pop = int(rf_clf.predict(X_row)[0])
        except Exception:
            pred_pop = 1 if pred_rating > 4.0 else 0
        popularity_text = "🔥 Popular" if pred_pop==1 else "ℹ️ Less Popular"

        # fetch cover image (cached)
        cover_img = fetch_cover_image(df.iloc[idx]['title'], df.iloc[idx]['authors'])
        # Card layout: left image, right metrics
        c1, c2 = st.columns([1,2])
        with c1:
            if cover_img:
                st.image(cover_img, width=160, caption=f"{df.iloc[idx]['title']}")
            else:
                st.image(Image.new("RGB", (160,240), color=(230,230,230)), width=160)
        with c2:
            st.markdown(f"### {df.iloc[idx]['title']}")
            st.markdown(f"**Author(s):** {df.iloc[idx]['authors']}")
            st.markdown(f"**Publisher:** {df.iloc[idx].get('publisher','-')} — **Year:** {int(df.iloc[idx].get('publication_year',np.nan)) if pd.notna(df.iloc[idx].get('publication_year')) else '-'}")
            st.markdown(f"**Predicted Rating:** <span style='font-size:20px'>{pred_rating:.2f} / 5</span>", unsafe_allow_html=True)
            st.markdown(f"**Predicted Popularity:** {popularity_text}")
            st.write(f"**Actual Avg Rating:** {df.iloc[idx]['average_rating']:.2f}  •  **Ratings Count:** {int(df.iloc[idx]['ratings_count'])}")

        st.markdown("---")
        # Recommendations with covers
        st.subheader("Recommended Books")
        recs = recommend_titles_by_index(idx, top_n=6)
        # show as cards (3 per row)
        cards = []
        for i, rec in enumerate(recs):
            t = rec['title']; a = rec['authors']; score = rec['score']; rating = rec['rating']
            img = fetch_cover_image(t, a)
            cards.append((t,a,score,rating,img))
        # display cards
        cols = st.columns(3)
        for i, (t,a,score,rating,img) in enumerate(cards):
            col = cols[i%3]
            with col:
                st.markdown("<div style='border-radius:8px; padding:8px; box-shadow: 0 3px 8px rgba(0,0,0,0.12);'>", unsafe_allow_html=True)
                if img:
                    st.image(img, width=140)
                else:
                    st.image(Image.new("RGB", (140,210), color=(240,240,240)), width=140)
                st.markdown(f"**{t}**")
                st.markdown(f"_{a}_")
                st.markdown(f"⭐ {rating:.2f}  •  similarity {score:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Sidebar: Charts & Insights
# -------------------------
st.sidebar.title("Dashboard • Insights")
# Top authors by influence
top_auth = df.groupby('authors')['author_influence'].mean().nlargest(12).reset_index()
fig_auth = px.bar(top_auth, x='author_influence', y='authors', orientation='h', labels={'author_influence':'Influence','authors':'Author'}, width=320, height=320)
st.sidebar.subheader("Top Authors by Influence")
st.sidebar.plotly_chart(fig_auth, use_container_width=True)

# Top publishers
top_pub = df.groupby('publisher')['publisher_influence'].mean().nlargest(12).reset_index()
fig_pub = px.bar(top_pub, x='publisher_influence', y='publisher', orientation='h', labels={'publisher_influence':'Publisher Influence','publisher':'Publisher'}, width=320, height=320)
st.sidebar.subheader("Top Publishers")
st.sidebar.plotly_chart(fig_pub, use_container_width=True)

# Rating distribution
fig_hist = px.histogram(df, x='average_rating', nbins=20, title="Rating Distribution", width=320, height=220)
st.sidebar.plotly_chart(fig_hist, use_container_width=True)

# Popularity breakdown
pop_counts = df['average_rating'].apply(lambda x: 1 if x>4 else 0).value_counts().reindex([1,0]).fillna(0)
fig_pie = go.Figure(data=[go.Pie(labels=['Popular','Not Popular'], values=[pop_counts.get(1,0), pop_counts.get(0,0)], hole=.35)])
fig_pie.update_layout(width=320, height=240, margin=dict(l=10,r=10,t=20,b=10))
st.sidebar.subheader("Popularity")
st.sidebar.plotly_chart(fig_pie, use_container_width=True)

# Feature importances (regressor)
fi = get_feature_importances(rf_reg, FEATURES)
fig_fi = px.bar(fi, x='importance', y='feature', orientation='h', labels={'importance':'Importance','feature':'Feature'}, width=320, height=240)
st.sidebar.subheader("Feature Importance")
st.sidebar.plotly_chart(fig_fi, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("📥 Download cleaned dataset")
@st.cache_data
def convert_df_to_csv(df_):
    return df_.to_csv(index=False).encode('utf-8')
csv_bytes = convert_df_to_csv(df)
st.sidebar.download_button("Download CSV", data=csv_bytes, file_name="books_cleaned_featured.csv", mime="text/csv")

# -------------------------
# Main: Explore table & filters
# -------------------------
# -------------------------
# Main: Explore Books Section (Fixed)
# -------------------------
st.markdown("---")
st.subheader("🔍 Explore Books")

# Ensure 'num_pages' is numeric
df['num_pages'] = pd.to_numeric(df['num_pages'], errors='coerce').fillna(0)

# Safe min/max for slider
min_pages_val = int(df['num_pages'].min())
max_pages_val = int(df['num_pages'].max() if df['num_pages'].max() > 0 else 1000)

# Streamlit slider for filtering pages
min_pages, max_pages = st.sidebar.slider(
    "Pages range",
    min_pages_val,
    max_pages_val,
    (min_pages_val, max_pages_val)
)

# Optional: Search by title
search_input = st.text_input("Search by book title (optional)")

# -------------------------
# Apply filters
# -------------------------
df_explore = df.copy()
if lang_filter:
    df_explore = df_explore[df_explore['language_code'].isin(lang_filter)]
if pub_filter != "All":
    df_explore = df_explore[df_explore['publisher'] == pub_filter]
df_explore = df_explore[
    (df_explore['num_pages'] >= min_pages) & (df_explore['num_pages'] <= max_pages)
]
if search_input.strip():
    df_explore = df_explore[df_explore['title'].str.contains(search_input, case=False, na=False)]

# Sort by rating descending
df_explore = df_explore.sort_values(by='average_rating', ascending=False).reset_index(drop=True)

# -------------------------
# Display books with thumbnails (top 50)
# -------------------------
def get_cover_image_safe(title, author):
    """Return PIL.Image for cover; placeholder if fails."""
    try:
        url = fetch_cover_google(title, author) or fetch_cover_openlibrary(title)
        if not url:
            url = "https://via.placeholder.com/60x90.png?text=No+Cover"
        response = requests.get(url, timeout=3)
        img = Image.open(BytesIO(response.content))
        img.thumbnail((60, 90))
        return img
    except:
        return Image.new("RGB", (60, 90), color=(230, 230, 230))

for idx, row in df_explore.head(15).iterrows():
    cols = st.columns([1, 4, 1, 1, 1])
    with cols[0]:
        st.image(get_cover_image_safe(row['title'], row['authors']))
    with cols[1]:
        st.markdown(f"**{row['title']}** by {row['authors']}")
    with cols[2]:
        st.write(f"⭐ {row['average_rating']:.2f}")
    with cols[3]:
        st.write(f"📝 {int(row['ratings_count'])}")
    with cols[4]:
        st.write(f"💬 {int(row['text_reviews_count'])}")

st.markdown("---")
st.markdown("Built with ❤️ • Data: Goodreads-derived dataset • Covers: Google Books / Open Library")
