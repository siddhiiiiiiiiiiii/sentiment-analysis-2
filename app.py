import html, io, os, re, random, sqlite3, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import requests
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report)

st.set_page_config(page_title="Social Media Sentiment Analyzer", page_icon="💬", layout="wide")

LABELS = ["Positive", "Neutral", "Negative"]
COLORS = {"Positive": "#4CAF50", "Neutral": "#9E9E9E", "Negative": "#F44336"}
ICONS = {"Positive": "😊", "Neutral": "😐", "Negative": "😠"}

# ---------------------------------------------------------------- preprocessing
try:
    STOP = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP = set(stopwords.words("english"))
STOP -= {"no", "nor", "not", "very", "never"}   # keep negations
stemmer = PorterStemmer()

def clean(text):
    text = str(text).lower().replace("\u2019", "'")
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b|\bcannot\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)            # don't, didn't, isn't ... -> "do not", "did not", "is not"
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(stemmer.stem(w) for w in text.split() if w not in STOP and len(w) > 1)

# ---------------------------------------------------------------- data
def make_sample_dataset():
    random.seed(42)
    pos_s = ["Absolutely loved", "Really happy with", "Such an amazing", "Totally recommend", "Best purchase ever:", "Super impressed by", "Great quality,", "Fantastic", "I love", "Excellent"]
    neg_s = ["Very poor", "Totally disappointed with", "Worst", "Terrible", "Do not buy", "Really unhappy with", "Awful", "Waste of money,", "I hate", "Horrible"]
    neu_s = ["It is okay,", "Average", "Received", "Just bought", "Not sure about", "It works as described,", "Using it since", "Delivered on", "Standard", "Neither good nor bad,"]
    things = ["this product", "the service", "the delivery", "the app", "the phone", "the update", "customer support", "the new design", "this brand", "the packaging"]
    pos_e = ["!", " - highly recommended!", ", works perfectly!", ", so good #happy", " :) great value", ", will buy again!", " https://t.co/xyz", " @brand thank you!!", ", five stars", ", exceeded expectations"]
    neg_e = ["!", " - never again.", ", stopped working in a day", ", so bad #fail", " :( total waste", ", will not buy again", " https://t.co/abc", " @brand fix this!!", ", one star", ", worst experience"]
    neu_e = [".", ", nothing special.", ", will see how it goes.", " #review", " https://t.co/def", ", as expected.", " @brand received it.", ", it is fine I guess.", ", no complaints no praise.", ", arrived on time."]
    rows = []
    for label, s, e, n in [("Positive", pos_s, pos_e, 130), ("Negative", neg_s, neg_e, 100), ("Neutral", neu_s, neu_e, 90)]:
        rows += [(f"{random.choice(s)} {random.choice(things)}{random.choice(e)}", label) for _ in range(n)]
    return pd.DataFrame(rows, columns=["comment", "sentiment"]).sample(frac=1, random_state=1).reset_index(drop=True)

def rating_to_sentiment(r):
    return "Negative" if r <= 2 else ("Neutral" if r == 3 else "Positive")

TEXT_KEYS = ["reviews.text", "reviewtext", "review_text", "review", "reviews", "text", "content", "summary", "comment"]
RATE_KEYS = ["reviews.rating", "overall", "rating", "rate", "stars", "score"]

def detect(raw, keys):
    cols = {c.lower().strip(): c for c in raw.columns}
    return next((cols[k] for k in keys if k in cols), None)

def guess_columns(raw):
    """For files without a header row: the rating column holds only 1-5, the text column has the longest strings."""
    rate = None
    for c in raw.columns:
        v = pd.to_numeric(raw[c], errors="coerce")
        if v.notna().mean() > 0.95 and v.dropna().between(1, 5).all() and v.nunique() >= 2:
            rate = c; break
    texts = [c for c in raw.columns if not pd.api.types.is_numeric_dtype(raw[c])]
    text = max(texts, key=lambda c: raw[c].astype(str).str.len().mean()) if texts else None
    return text, rate

def load_reviews(file, per_class=3000):
    def read(**kw):
        if hasattr(file, "seek"):
            file.seek(0)
        return pd.read_csv(file, encoding="utf-8", encoding_errors="ignore", **kw)
    raw = read()
    tc, rc = detect(raw, TEXT_KEYS), detect(raw, RATE_KEYS)
    if tc is None or rc is None:            # the file may have no header row: read it again without one
        raw = read(header=None)
        raw.columns = [f"column_{i}" for i in range(raw.shape[1])]
        tc, rc = guess_columns(raw)
    if tc is None or rc is None:
        raise ValueError("Could not find a review-text column and a 1-5 star-rating column. "
                         f"First columns seen: {list(raw.columns)[:6]}")
    d = pd.DataFrame({"comment": raw[tc].astype(str), "rating": pd.to_numeric(raw[rc], errors="coerce")}).dropna()
    d["sentiment"] = d["rating"].round().astype(int).apply(rating_to_sentiment)
    d = pd.concat([g.sample(min(len(g), per_class), random_state=42) for _, g in d.groupby("sentiment")])
    return d[["comment", "sentiment"]].sample(frac=1, random_state=1).reset_index(drop=True)

@st.cache_data(show_spinner="Loading dataset...")
def load_reviews_cached(src, per_class, mtime=0):
    """src is a file path (str) or the raw bytes of an uploaded file."""
    return load_reviews(io.BytesIO(src) if isinstance(src, bytes) else src, per_class)

# ---------------------------------------------------------------- training
@st.cache_resource(show_spinner="Training models...")
def train(df: pd.DataFrame):
    d = df.copy()
    d["clean"] = d["comment"].apply(clean)
    Xtr_t, Xte_t, ytr, yte = train_test_split(d["clean"], d["sentiment"], test_size=0.2,
                                              random_state=42, stratify=d["sentiment"])
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=200000)
    Xtr, Xte = tfidf.fit_transform(Xtr_t), tfidf.transform(Xte_t)
    models = {"Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
              "Naive Bayes": MultinomialNB(fit_prior=False)}
    results, preds = {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte); preds[name] = p
        results[name] = {"Accuracy": accuracy_score(yte, p),
                         "Precision": precision_score(yte, p, average="weighted", zero_division=0),
                         "Recall": recall_score(yte, p, average="weighted"),
                         "F1-score": f1_score(yte, p, average="weighted")}
    res = pd.DataFrame(results).T.round(3)
    return {"tfidf": tfidf, "models": models, "results": res, "preds": preds,
            "y_test": yte, "best": res["F1-score"].idxmax(), "n": len(d)}

def proba(model, tfidf, text):
    p = model.predict_proba(tfidf.transform([clean(text)]))[0]
    return {c: float(v) for c, v in zip(model.classes_, p)}

# ---------------------------------------------------------------- live pipeline (ETL)
DB_PATH = "sentiment.db"
DB_COLS = ["uid", "source", "author", "created_at", "text", "clean", "label_model", "conf_model",
           "vader_score", "label_vader", "label", "fetched_at"]
FETCH_COLS = ["uid", "source", "author", "created_at", "text"]

@st.cache_resource
def get_vader():
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()

@st.cache_resource
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS comments (
        uid TEXT PRIMARY KEY, source TEXT, author TEXT, created_at TEXT, text TEXT, clean TEXT,
        label_model TEXT, conf_model REAL, vader_score REAL, label_vader TEXT, label TEXT, fetched_at TEXT)""")
    conn.commit()
    return conn

def strip_html(t):
    return html.unescape(re.sub(r"<[^>]+>", " ", t or "")).strip()

def fetch_hackernews(query, n):
    r = requests.get("https://hn.algolia.com/api/v1/search_by_date",
                     params={"query": query, "tags": "comment", "hitsPerPage": n}, timeout=20)
    r.raise_for_status()
    rows = []
    for h in r.json().get("hits", []):
        text = strip_html(h.get("comment_text"))
        if text:
            rows.append({"uid": "hn_" + str(h["objectID"]), "source": "Hacker News", "author": h.get("author"),
                         "created_at": h.get("created_at"), "text": text})
    return pd.DataFrame(rows, columns=FETCH_COLS)

def youtube_id(v):
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", v)
    return m.group(1) if m else v.strip()

def fetch_youtube(video, key, n):
    if not key:
        raise ValueError("YouTube API key missing. Paste it in the box or add YOUTUBE_API_KEY to the app secrets.")
    r = requests.get("https://www.googleapis.com/youtube/v3/commentThreads",
                     params={"part": "snippet", "videoId": youtube_id(video), "maxResults": min(n, 100),
                             "order": "time", "textFormat": "plainText", "key": key}, timeout=20)
    r.raise_for_status()
    rows = []
    for it in r.json().get("items", []):
        sn = it["snippet"]["topLevelComment"]["snippet"]
        rows.append({"uid": "yt_" + it["id"], "source": "YouTube", "author": sn.get("authorDisplayName"),
                     "created_at": sn.get("publishedAt"), "text": sn.get("textDisplay", "").strip()})
    return pd.DataFrame(rows, columns=FETCH_COLS)

def vader_label(score):
    return "Positive" if score >= 0.05 else ("Negative" if score <= -0.05 else "Neutral")

def analyze_live(df, model, tfidf, engine):
    """Transform step: clean text, predict with the trained model and with VADER."""
    out = df.copy()
    out["created_at"] = pd.to_datetime(out["created_at"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    out["clean"] = out["text"].apply(clean)
    probs = model.predict_proba(tfidf.transform(out["clean"]))
    out["label_model"] = model.classes_[probs.argmax(axis=1)]
    out["conf_model"] = probs.max(axis=1).round(3)
    empty = out["clean"].str.strip() == ""
    out.loc[empty, ["label_model", "conf_model"]] = ["Neutral", 0.0]
    vader = get_vader()
    out["vader_score"] = out["text"].apply(lambda t: vader.polarity_scores(t)["compound"])
    out["label_vader"] = out["vader_score"].apply(vader_label)
    out["label"] = out["label_model"] if engine == "TF-IDF model" else out["label_vader"]
    out["fetched_at"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")
    return out

def run_once(src, term, n, key, model, tfidf, engine):
    """One ETL run: extract -> transform -> load. Returns (new_comments, error_message)."""
    conn = get_db()
    try:
        raw = fetch_hackernews(term, n) if src == "Hacker News" else fetch_youtube(term, key, n)
    except Exception as e:
        return 0, str(e)
    known = {r[0] for r in conn.execute("SELECT uid FROM comments")}
    raw = raw[~raw["uid"].isin(known)]
    if raw.empty:
        return 0, None
    out = analyze_live(raw, model, tfidf, engine)
    conn.executemany(f"INSERT OR IGNORE INTO comments ({','.join(DB_COLS)}) VALUES ({','.join('?' * len(DB_COLS))})",
                     out[DB_COLS].astype(object).where(out[DB_COLS].notna(), None).values.tolist())
    conn.commit()
    return len(out), None

def draw_pie(ax, dist):
    nz = dist[dist > 0]
    if len(nz):
        ax.pie(nz.values, labels=nz.index, autopct="%1.1f%%", startangle=90, colors=[COLORS[l] for l in nz.index])

def read_csv_smart(file):
    """Read a CSV; if it has no header row (first line is data), re-read it with generic column names."""
    def read(**kw):
        file.seek(0)
        return pd.read_csv(file, encoding="utf-8", encoding_errors="ignore", **kw)
    raw = read()
    named = detect(raw, TEXT_KEYS) is not None or detect(raw, RATE_KEYS) is not None
    looks_like_data = any(str(c).strip().isdigit() or len(str(c)) > 40 for c in raw.columns)
    if not named and looks_like_data:
        raw = read(header=None)
        raw.columns = [f"column_{i}" for i in range(raw.shape[1])]
    return raw

def time_trend(df):
    span = df["created_at"].max() - df["created_at"].min()
    freq = ("1min" if span < pd.Timedelta(hours=1) else "10min" if span < pd.Timedelta(hours=6)
            else "1h" if span < pd.Timedelta(days=2) else "1D")
    counts = (df.groupby([pd.Grouper(key="created_at", freq=freq), "label"]).size()
                .unstack(fill_value=0).reindex(columns=LABELS, fill_value=0))
    return counts.div(counts.sum(axis=1), axis=0) * 100, freq

def render_dashboard():
    conn = get_db()
    df = pd.read_sql("SELECT * FROM comments", conn)
    if df.empty:
        st.info("The warehouse is empty. Click **Fetch now** to load the first comments.")
        return
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    dist = df["label"].value_counts().reindex(LABELS).fillna(0)
    m = st.columns(4)
    m[0].metric("Total comments", len(df))
    for col, l in zip(m[1:], LABELS):
        col.metric(f"{ICONS[l]} {l}", int(dist[l]))
    c1, c2, c3 = st.columns(3)
    fig, ax = plt.subplots(figsize=(4, 3.2))
    ax.bar(dist.index, dist.values, color=[COLORS[l] for l in dist.index]); ax.set_ylabel("Comments")
    c1.pyplot(fig)
    fig, ax = plt.subplots(figsize=(4, 3.2))
    draw_pie(ax, dist)
    c2.pyplot(fig)
    pct, freq = time_trend(df)
    fig, ax = plt.subplots(figsize=(4, 3.2))
    pct.plot(ax=ax, marker="o", color=[COLORS[l] for l in LABELS]); ax.set_ylabel("% of comments")
    ax.set_xlabel(""); ax.set_title(f"per {freq}", fontsize=9); ax.legend(fontsize=7)
    c3.pyplot(fig)
    st.subheader("Latest comments")
    latest = df.sort_values("created_at", ascending=False).head(15).copy()
    latest["text"] = latest["text"].str.slice(0, 140)
    st.dataframe(latest[["created_at", "source", "label", "text"]], hide_index=True)
    with st.expander("📑 Warehouse reports (SQL)"):
        q = {"Sentiment summary": """SELECT label, COUNT(*) AS comments,
                 ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM comments), 1) AS percent,
                 ROUND(AVG(vader_score), 3) AS avg_vader_score FROM comments GROUP BY label ORDER BY comments DESC""",
             "Sentiment per hour": """SELECT strftime('%Y-%m-%d %H:00', created_at) AS hour, label, COUNT(*) AS comments
                 FROM comments GROUP BY hour, label ORDER BY hour""",
             "Agreement: TF-IDF model vs VADER": """SELECT COUNT(*) AS comments,
                 ROUND(100.0 * SUM(label_model = label_vader) / COUNT(*), 1) AS agreement_percent FROM comments"""}
        for title, sql in q.items():
            st.markdown(f"**{title}**")
            st.dataframe(pd.read_sql(sql, conn), hide_index=True)
    st.download_button("⬇️ Download warehouse as CSV", df.to_csv(index=False).encode(),
                       "sentiment_warehouse.csv", "text/csv")

# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Settings")
SOURCES = ["Generated sample", "Upload reviews CSV", "Local file path"]
source = st.sidebar.radio("Training data", SOURCES, index=2 if os.path.isfile("reviews.csv") else 0)
per_class = st.sidebar.slider("Max reviews per class (balanced)", 500, 50000, 5000, 500,
                              help="Review data is mostly 5-star, so each class is capped to keep the model fair.")
df = None
if source == "Upload reviews CSV":
    up = st.sidebar.file_uploader("CSV with review text + star rating", type="csv")
    if up is not None:
        try:
            df = load_reviews_cached(up.getvalue(), per_class)
        except Exception as e:
            st.sidebar.error(str(e))
    if df is None:
        st.info("Upload a reviews CSV in the sidebar (1-2★ Negative, 3★ Neutral, 4-5★ Positive), "
                "or switch to the generated sample.")
        st.stop()
elif source == "Local file path":
    path = st.sidebar.text_input("Path to reviews CSV on this computer", "reviews.csv")
    if os.path.isfile(path):
        try:
            df = load_reviews_cached(path, per_class, os.path.getmtime(path))
        except Exception as e:
            st.sidebar.error(str(e))
    elif path:
        st.sidebar.error(f"File not found: {path}")
    if df is None:
        st.info("Enter the path to a reviews CSV (text + star rating) in the sidebar, or switch to the generated sample.")
        st.stop()
else:
    df = make_sample_dataset()

art = train(df)
choice = st.sidebar.selectbox("Model", ["Best (auto)"] + list(art["models"]))
model_name = art["best"] if choice == "Best (auto)" else choice
model, tfidf = art["models"][model_name], art["tfidf"]
st.sidebar.caption(f"Training rows: {art['n']}  \nActive model: **{model_name}**")
if df is not None and source != "Generated sample":
    st.sidebar.caption("Class counts: " + ", ".join(f"{k}: {v}" for k, v in df.sentiment.value_counts().items()))
if source == "Generated sample":
    st.sidebar.warning("The generated sample is very simple, so accuracy is near 100%. Use real reviews for realistic results.")

# ---------------------------------------------------------------- main
st.title("💬 Social Media Sentiment Analyzer")
st.caption("Data Warehousing and Mining – Mini Project · TF-IDF + Logistic Regression / Naive Bayes")
tab1, tab2, tab3, tab4 = st.tabs(["📝 Analyze a comment", "📂 Batch CSV", "📊 Model performance", "📡 Live feed"])

# ---- Tab 1
with tab1:
    EXAMPLES = ["", "Absolutely loved this product, works perfectly!", "It is okay, nothing special.",
                "Very poor experience, never again.", "The delivery was late and the box was damaged"]
    if "history" not in st.session_state:
        st.session_state.history = []
    st.selectbox("Try an example", EXAMPLES, key="example",
                 on_change=lambda: st.session_state.update(comment=st.session_state.example))
    text = st.text_area("Comment / review", key="comment", height=120,
                        placeholder="Type or paste a comment here...")
    c1, c2, _ = st.columns([1, 1, 6])
    go = c1.button("Analyze", type="primary")
    if c2.button("Clear history"):
        st.session_state.history = []
    if go:
        if not text.strip():
            st.warning("Please enter a comment first.")
        elif not clean(text):
            st.warning("No meaningful words found after cleaning - try a longer comment.")
        else:
            probs = proba(model, tfidf, text)
            label = max(probs, key=probs.get)
            st.session_state.history.insert(0, {"Comment": text[:80] + ("..." if len(text) > 80 else ""),
                                                "Sentiment": label, "Confidence %": round(probs[label] * 100, 1)})
            st.markdown(f"### {ICONS[label]} <span style='color:{COLORS[label]}'>{label}</span> "
                        f"<small>({probs[label]*100:.1f}% confidence)</small>", unsafe_allow_html=True)
            for l in LABELS:
                st.progress(probs.get(l, 0.0), text=f"{l}: {probs.get(l, 0.0)*100:.1f}%")
    if st.session_state.history:
        st.subheader("History")
        st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True)

# ---- Tab 2
with tab2:
    st.write("Upload a CSV of comments to classify them all at once. Files without a header row are handled too.")
    bf = st.file_uploader("Comments CSV", type="csv", key="batch")
    if bf is not None:
        raw = read_csv_smart(bf)
        g_text, g_rate = guess_columns(raw)
        text_guess = detect(raw, TEXT_KEYS) or g_text or raw.columns[0]
        rate_guess = detect(raw, RATE_KEYS) or g_rate
        cols = list(raw.columns)
        c1, c2 = st.columns(2)
        col = c1.selectbox("Text column", cols, index=cols.index(text_guess))
        rate_opts = ["(none)"] + cols
        rcol = c2.selectbox("Star-rating column (optional, to check accuracy)", rate_opts,
                            index=rate_opts.index(rate_guess) if rate_guess in rate_opts else 0)
        if st.button("Classify all", type="primary"):
            out = raw.copy()
            out["predicted_sentiment"] = model.predict(tfidf.transform(out[col].astype(str).apply(clean)))
            dist = out["predicted_sentiment"].value_counts().reindex(LABELS).fillna(0)
            a, b = st.columns(2)
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            ax.bar(dist.index, dist.values, color=[COLORS[l] for l in dist.index]); ax.set_ylabel("Comments")
            a.pyplot(fig)
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            draw_pie(ax, dist)
            b.pyplot(fig)
            if rcol != "(none)":
                stars = pd.to_numeric(out[rcol], errors="coerce")
                ok = stars.between(1, 5)
                if ok.sum() > 0:
                    actual = stars[ok].round().astype(int).apply(rating_to_sentiment)
                    pred = out.loc[ok, "predicted_sentiment"]
                    st.metric("Agreement with the star ratings", f"{(actual == pred).mean() * 100:.1f}%")
                    st.write("Actual (from stars) vs predicted")
                    st.dataframe(pd.crosstab(actual.rename("actual"), pred.rename("predicted"))
                                   .reindex(index=LABELS, columns=LABELS, fill_value=0))
                    st.caption("If this file was also used for training, this is not a fair test: the model has already seen these reviews.")
                else:
                    st.info("The chosen rating column has no values between 1 and 5.")
            st.dataframe(out)
            st.download_button("⬇️ Download results", out.to_csv(index=False).encode(),
                               "sentiment_results.csv", "text/csv")

# ---- Tab 3
with tab3:
    st.subheader("Model comparison")
    st.dataframe(art["results"])
    st.bar_chart(art["results"])
    st.caption(f"Best model by F1-score: **{art['best']}**")
    show = st.selectbox("Details for", list(art["models"]), index=list(art["models"]).index(model_name))
    y_test, p = art["y_test"], art["preds"][show]
    a, b = st.columns(2)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, p, labels=LABELS), display_labels=LABELS).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion matrix – {show}")
    a.pyplot(fig)
    b.code(classification_report(y_test, p, labels=LABELS, zero_division=0))


# ---- Tab 4: live data pipeline
with tab4:
    st.write("Fetch **live comments** from a website, clean them, predict their sentiment and store them in a "
             "small SQLite data warehouse (**Extract → Transform → Load**).")
    a, b = st.columns(2)
    live_src = a.selectbox("Source", ["Hacker News", "YouTube"], key="live_src")
    if live_src == "Hacker News":
        term = b.text_input("Search word", "iphone", key="live_term_hn")
    else:
        term = b.text_input("Video link or ID", "dQw4w9WgXcQ", key="live_term_yt")
    yt_key = ""
    if live_src == "YouTube":
        try:
            yt_key = st.secrets.get("YOUTUBE_API_KEY", "")
        except Exception:
            yt_key = ""
        if not yt_key:
            yt_key = st.text_input("YouTube Data API key", type="password", key="live_yt_key",
                                   help="Free key from Google Cloud Console (YouTube Data API v3).")
    c, d = st.columns(2)
    n_fetch = c.slider("Comments per fetch", 10, 100, 50, 10, key="live_n")
    engine = d.radio("Final label comes from", ["VADER", "TF-IDF model"], horizontal=True, key="live_engine",
                     index=0 if source == "Generated sample" else 1,
                     help="Both predictions are always stored. VADER is a ready-made tool for social text. "
                          "The TF-IDF model is the one trained in this project.")
    if source == "Generated sample" and engine == "TF-IDF model":
        st.warning("The TF-IDF model is trained on the simple generated sample, so it will be weak on real comments. "
                   "Train it on a real reviews CSV (sidebar) or use VADER.")
    e, f, g = st.columns([1, 1, 3])
    do_fetch = e.button("Fetch now", type="primary", key="live_fetch")
    if f.button("Clear warehouse", key="live_clear"):
        get_db().execute("DELETE FROM comments"); get_db().commit()
        st.success("Warehouse cleared.")
    auto = g.checkbox("Auto-refresh (checks for new comments)", key="live_auto")
    secs = g.slider("Seconds between refreshes", 15, 300, 30, 15, key="live_secs") if auto else None

    if do_fetch:
        with st.spinner("Fetching and analyzing..."):
            n_new, err = run_once(live_src, term, n_fetch, yt_key, model, tfidf, engine)
        if err:
            st.error(f"Fetch failed: {err}")
        else:
            st.success(f"{n_new} new comments stored." if n_new else "No new comments since the last fetch.")

    def live_panel():
        if auto:
            n_new, err = run_once(live_src, term, n_fetch, yt_key, model, tfidf, engine)
            st.caption(f"Auto-refresh {time.strftime('%H:%M:%S')}: " + (f"error - {err}" if err else f"{n_new} new comments"))
        render_dashboard()

    if auto and hasattr(st, "fragment"):
        st.fragment(run_every=secs)(live_panel)()
    else:
        live_panel()
    st.caption("The model is trained on product reviews, so on forum or video comments expect mistakes. "
               "The warehouse file (sentiment.db) is temporary on free cloud hosting.")
