import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import os

# -----------------------------
# Load and preprocess dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join("data", "netflix_titles.csv"))
    df.dropna(subset=["title", "listed_in", "description"], inplace=True)
    return df

df = load_data()

# -----------------------------
# Feature extraction for similarity
# -----------------------------
@st.cache_data
def compute_similarity(data):
    vectorizer = CountVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(data['listed_in'] + " " + data['description'])
    similarity = cosine_similarity(matrix)
    return similarity

similarity = compute_similarity(df)

# -----------------------------
# Recommendation Function
# -----------------------------
def get_recommendations(title, n=5):
    title = title.lower()
    matches = df[df['title'].str.lower().str.contains(title)]
    if matches.empty:
        return None
    idx = matches.index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in scores[1:n+1]]
    return df.iloc[top_indices]

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="StreamFlix Recommendations", layout="wide")

# -----------------------------
# Custom CSS (no glow, clean Netflix red)
# -----------------------------
st.markdown("""
<style>
.recommend-card {
    background-color:#1a1a1a;
    border-radius:15px;
    padding:20px;
    margin-bottom:15px;
    box-shadow: 0 0 15px rgba(255,0,0,0.15);
    transition: all 0.3s ease;
    border: 1px solid rgba(255,255,255,0.05);
}
.recommend-card:hover {
    transform: scale(1.02);
    box-shadow: 0 0 25px rgba(229,9,20,0.4);
    border: 1px solid rgba(229,9,20,0.6);
    cursor: pointer;
}
.genre-pill {
    display:inline-block;
    background-color: rgba(229,9,20,0.15);
    color:#E50914;
    border:1px solid rgba(229,9,20,0.4);
    border-radius:12px;
    padding:3px 8px;
    font-size:12px;
    margin:2px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header Section (no glow animation)
# -----------------------------
st.markdown(
    """
    <div style='text-align:center; margin-top:-40px;'>
        <h1 style='
            font-size:48px;
            font-weight:bold;
            color:#E50914;
            display:inline-flex;
            align-items:center;
            justify-content:center;
            gap:12px;
        '>
            🍿 StreamFlix Recommendations
        </h1>
        <p style='font-size:20px; color:#ddd; margin-top:12px;'>
            Your personalized movie & TV show recommendations!
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Add spacing below title
st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.title("Filters")

type_filter = st.sidebar.selectbox("Type", ["All"] + sorted(df["type"].dropna().unique().tolist()))
unique_genres = sorted(set(g.strip() for s in df["listed_in"].dropna() for g in s.split(",")))
genre_filter = st.sidebar.multiselect("Genre(s)", unique_genres)
unique_countries = sorted(set(c.strip() for s in df["country"].dropna() for c in str(s).split(",")))
country_filter = st.sidebar.selectbox("Country", ["All"] + unique_countries)

# -----------------------------
# Search Section
# -----------------------------
st.markdown("### Search for a show/movie and get similar recommendations!")
user_input = st.text_input("Enter show/movie name:")

# -----------------------------
# Recommendation Display
# -----------------------------
if user_input:
    recs = get_recommendations(user_input, n=10)
    if recs is not None:
        # Apply filters
        if type_filter != "All":
            recs = recs[recs["type"] == type_filter]
        if genre_filter:
            recs = recs[recs["listed_in"].apply(lambda x: any(g in x for g in genre_filter))]
        if country_filter != "All":
            recs = recs[recs["country"].apply(lambda x: country_filter in str(x))]

        if not recs.empty:
            st.markdown("## Recommended Shows/Movies: 🔁")
            for _, row in recs.iterrows():
                genres_html = " ".join([f"<span class='genre-pill'>{g.strip()}</span>" for g in str(row.get('listed_in', 'N/A')).split(",")])
                search_url = f"https://www.google.com/search?q={row['title'].replace(' ', '+')}+Netflix"
                st.markdown(
                    f"""
                    <a href='{search_url}' target='_blank' style='text-decoration:none;'>
                        <div class='recommend-card'>
                            <h3 style='color:#ff3333;'>{row['title']}</h3>
                            <p style='color:#f5c518; font-size:14px;'>
                                <strong>Type:</strong> {row['type']} |
                                <strong>Year:</strong> {row.get('release_year', 'N/A')}
                            </p>
                            <p style='color:#ddd;'>
                                <strong>Country:</strong> {row.get('country', 'Unknown')}
                            </p>
                            <div>{genres_html}</div>
                            <p style='color:#bbb; margin-top:8px;'><em>{row['description']}</em></p>
                        </div>w
                    </a>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("No matches found with selected filters.")
    else:
        st.error("No show/movie found! Try another title.")
else:
    st.info("🔍 Start by typing a movie or show name to see recommendations!")