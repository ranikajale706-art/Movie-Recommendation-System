import streamlit as st
import pandas as pd
import requests

from pathlib import Path
from io import BytesIO
from PIL import Image

from database import (
    register_user,
    login_user
)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# PROFESSIONAL DESIGN
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: #ffffff;
    color: #172033;
}

[data-testid="stSidebar"] {
    background: #f4f7fb;
    border-right: 1px solid #e4eaf2;
}

h1, h2, h3 {
    color: #142b4a !important;
}

.brand-card {
    background: linear-gradient(110deg, #102d50, #245b91);
    padding: 24px 28px;
    border-radius: 16px;
    color: white;
    margin-bottom: 22px;
}

.brand-card h1 {
    color: white !important;
    margin: 0;
    font-size: 2rem;
}

.brand-card p {
    color: #e5effb;
    margin: 6px 0 0 0;
}

div.stButton > button {
    border-radius: 9px;
}

[data-testid="stMetric"] {
    background: #f5f8fc;
    border: 1px solid #e4eaf2;
    padding: 14px;
    border-radius: 12px;
}

hr {
    border-color: #e6ebf2;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MOVIE DATASET
# --------------------------------------------------

DATA_FILE = Path(__file__).parent / "movies.csv"


@st.cache_data
def load_movies():

    if not DATA_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(DATA_FILE).fillna("")


movies = load_movies()

required_columns = [
    "title",
    "genre",
    "description",
    "poster"
]

if movies.empty:
    st.error(
        "movies.csv is missing or empty. "
        "Keep it in the same folder as app.py."
    )
    st.stop()


missing_columns = [
    column for column in required_columns
    if column not in movies.columns
]

if missing_columns:
    st.error(
        "Required columns missing from movies.csv: "
        + ", ".join(missing_columns)
    )
    st.stop()


if "language" not in movies.columns:
    movies["language"] = "English"


for column in [
    "title",
    "genre",
    "description",
    "poster",
    "language"
]:
    movies[column] = movies[column].astype(str)


# --------------------------------------------------
# MACHINE LEARNING MODEL
# CONTENT-BASED FILTERING
# --------------------------------------------------

movies["features"] = (
    movies["genre"] + " " + movies["description"]
)

vectorizer = CountVectorizer(stop_words="english")

try:

    feature_matrix = vectorizer.fit_transform(
        movies["features"]
    )

    similarity = cosine_similarity(feature_matrix)

except ValueError:

    similarity = None


# --------------------------------------------------
# LOAD MOVIE POSTERS
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def load_poster(url):

    if not url or not url.startswith(
        ("https://", "http://")
    ):
        return None

    try:

        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        response.raise_for_status()

        image = Image.open(
            BytesIO(response.content)
        )

        image.load()

        return image.copy()

    except Exception:

        return None


def show_poster(url, caption=None):

    image = load_poster(str(url).strip())

    if image is not None:

        st.image(
            image,
            use_container_width=True,
            caption=caption
        )

    else:

        st.markdown(
            """
            <div style='height:210px;
            background:#eef3f8;
            border-radius:10px;
            display:flex;
            align-items:center;
            justify-content:center;
            color:#55708c;
            font-size:42px;'>
            🎬
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption("Poster unavailable")


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

def brand_header(subtitle):

    st.markdown(
        f"""
        <div class="brand-card">
            <h1>🎬 Movie Recommendation System</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# SESSION MANAGEMENT
# --------------------------------------------------

if "user" not in st.session_state:
    st.session_state.user = None

if "recommendation_results" not in st.session_state:
    st.session_state.recommendation_results = []

if "recommendation_for" not in st.session_state:
    st.session_state.recommendation_for = None

if "movie_page" not in st.session_state:
    st.session_state.movie_page = 1


# --------------------------------------------------
# LOGIN AND REGISTRATION
# --------------------------------------------------

if st.session_state.user is None:

    brand_header(
        "Discover films you'll love. Find your next favourite."
    )

    left, center, right = st.columns([1, 1.4, 1])

    with center:

        st.subheader("Welcome")

        st.write(
            "Log in to explore personalized movie recommendations."
        )

        login_tab, register_tab = st.tabs(
            ["Login", "Create account"]
        )

        # LOGIN

        with login_tab:

            with st.form("login_form"):

                username = st.text_input(
                    "Username or email"
                )

                password = st.text_input(
                    "Password",
                    type="password"
                )

                submitted = st.form_submit_button(
                    "Login",
                    type="primary",
                    use_container_width=True
                )

                if submitted:

                    user = login_user(
                        username.strip(),
                        password
                    )

                    if user:

                        st.session_state.user = user

                        st.success(
                            "Login successful."
                        )

                        st.rerun()

                    else:

                        st.error(
                            "Invalid username/email or password."
                        )

        # REGISTRATION

        with register_tab:

            with st.form("register_form"):

                new_username = st.text_input(
                    "Username"
                )

                new_email = st.text_input(
                    "Email"
                )

                new_password = st.text_input(
                    "Create password",
                    type="password"
                )

                confirm_password = st.text_input(
                    "Confirm password",
                    type="password"
                )

                registered = st.form_submit_button(
                    "Create account",
                    type="primary",
                    use_container_width=True
                )

                if registered:

                    if (
                        not new_username.strip()
                        or not new_email.strip()
                        or not new_password
                    ):

                        st.error(
                            "Please fill in all fields."
                        )

                    elif (
                        "@" not in new_email
                        or "." not in new_email
                    ):

                        st.error(
                            "Enter a valid email address."
                        )

                    elif len(new_password) < 8:

                        st.error(
                            "Password must contain at least 8 characters."
                        )

                    elif new_password != confirm_password:

                        st.error(
                            "Passwords do not match."
                        )

                    else:

                        success, message = register_user(
                            new_username.strip(),
                            new_email.strip(),
                            new_password
                        )

                        if success:

                            st.success(
                                "Account created. Please log in."
                            )

                        else:

                            st.error(message)

    st.stop()


# --------------------------------------------------
# SIDEBAR AND USER PROFILE
# --------------------------------------------------

user = st.session_state.user

with st.sidebar:

    st.markdown(
        "## 🎬 Movie Recommendation System"
    )

    st.caption(
        "Your personal movie guide"
    )

    st.divider()

    st.markdown("### 👤 My Profile")

    st.write("**Username**")
    st.write(user["username"])

    st.write("**Email**")
    st.write(user["email"])

    st.divider()

    st.markdown("### Navigation")

    st.write("🎬 Discover Movies")

    st.divider()

    if st.button(
        "Log out",
        use_container_width=True
    ):

        st.session_state.user = None
        st.session_state.recommendation_results = []
        st.session_state.recommendation_for = None

        st.rerun()


# --------------------------------------------------
# DISCOVER PAGE
# --------------------------------------------------

brand_header(
    f"Welcome, {user['username']}! "
    "Discover movies you may enjoy."
)


# --------------------------------------------------
# MOVIE STATISTICS
# --------------------------------------------------

count_col, category_col, language_col = st.columns(3)

with count_col:

    st.metric(
        "Movies in library",
        len(movies)
    )

with category_col:

    st.metric(
        "Movie Categories",
        movies["genre"].nunique()
    )

with language_col:

    st.metric(
        "Languages",
        movies["language"].nunique()
    )


# --------------------------------------------------
# SEARCH AND FILTERS
# --------------------------------------------------

st.subheader("🔍 Find Your Movie")

search_col, category_col, language_col = st.columns(3)

with search_col:

    search = st.text_input(
        "Search by movie title",
        placeholder="e.g. Inception"
    )

with category_col:

    categories = sorted([
        category
        for category in movies["genre"].unique().tolist()
        if category
    ])

    selected_category = st.selectbox(
        "🎭 Filter by Movie Category",
        ["All Categories"] + categories
    )

with language_col:

    languages = sorted([
        language
        for language in movies["language"].unique().tolist()
        if language
    ])

    selected_language = st.selectbox(
        "🌐 Filter by language",
        ["All Languages"] + languages
    )


# --------------------------------------------------
# APPLY FILTERS
# --------------------------------------------------

filtered = movies.copy()

if selected_category != "All Categories":

    filtered = filtered[
        filtered["genre"] == selected_category
    ]

if selected_language != "All Languages":

    filtered = filtered[
        filtered["language"] == selected_language
    ]

if search.strip():

    filtered = filtered[
        filtered["title"].str.contains(
            search.strip(),
            case=False,
            na=False
        )
    ]


# --------------------------------------------------
# SELECT AND DISPLAY MOVIE
# --------------------------------------------------

if filtered.empty:

    st.warning(
        "No movies found. Try another search, "
        "movie category or language."
    )

else:

    st.caption(
        f"{len(filtered)} matching movie(s) found."
    )

    selected = st.selectbox(
        "🍿 Choose a movie",
        filtered["title"].tolist()
    )

    selected_index = movies.index[
        movies["title"] == selected
    ][0]

    selected_movie = movies.loc[selected_index]


    # MOVIE DETAILS

    preview_col, details_col = st.columns([1, 2])

    with preview_col:

        show_poster(
            selected_movie["poster"],
            selected
        )

    with details_col:

        st.markdown(
            f"### {selected_movie['title']}"
        )

        st.write(
            f"**Movie Category:** {selected_movie['genre']}"
        )

        st.write(
            f"**Language:** {selected_movie['language']}"
        )

        st.write(
            selected_movie["description"]
        )


    # --------------------------------------------------
    # ML-BASED MOVIE RECOMMENDATIONS
    # --------------------------------------------------

    if st.button(
        "✨ Recommend similar movies",
        type="primary"
    ):

        if similarity is None:

            st.warning(
                "Not enough movie descriptions "
                "to calculate recommendations."
            )

        else:

            scores = list(
                enumerate(similarity[selected_index])
            )

            scores.sort(
                key=lambda item: item[1],
                reverse=True
            )

            results = [
                (idx, score)
                for idx, score in scores
                if idx != selected_index
            ][:5]

            st.session_state.recommendation_results = [
                (int(idx), float(score))
                for idx, score in results
            ]

            st.session_state.recommendation_for = selected


    # DISPLAY RECOMMENDATIONS

    if (
        st.session_state.recommendation_results
        and st.session_state.recommendation_for == selected
    ):

        st.divider()

        st.subheader(
            f"🎯 Similar movies to {selected}"
        )

        result_cols = st.columns(5)

        for col, (idx, score) in zip(
            result_cols,
            st.session_state.recommendation_results
        ):

            movie = movies.iloc[idx]

            with col:

                show_poster(
                    movie["poster"],
                    movie["title"]
                )

                st.markdown(
                    f"**{movie['title']}**"
                )

                st.caption(
                    f"Movie Category: {movie['genre']} • "
                    f"{movie['language']}"
                )

                st.write(
                    f"Similarity: {round(score * 100)}%"
                )


# --------------------------------------------------
# EXPLORE ALL MOVIES
# --------------------------------------------------

st.divider()

st.subheader("🎞️ Explore All Movies")

st.write(
    "Browse movies from different categories and languages."
)

all_movies = movies.copy()

if selected_language != "All Languages":

    all_movies = all_movies[
        all_movies["language"] == selected_language
    ]


# --------------------------------------------------
# PAGINATION
# --------------------------------------------------

movies_per_page = 10

total_movies = len(all_movies)

total_pages = max(
    1,
    (total_movies + movies_per_page - 1)
    // movies_per_page
)

st.session_state.movie_page = min(
    st.session_state.movie_page,
    total_pages
)

start = (
    st.session_state.movie_page - 1
) * movies_per_page

page_movies = all_movies.iloc[
    start:start + movies_per_page
]

st.caption(
    f"{total_movies} movies available"
)


# --------------------------------------------------
# DISPLAY ALL MOVIES
# --------------------------------------------------

for _, movie in page_movies.iterrows():

    poster_col, info_col = st.columns(
        [1, 4]
    )

    with poster_col:

        show_poster(
            movie["poster"],
            movie["title"]
        )

    with info_col:

        st.markdown(
            f"### {movie['title']}"
        )

        st.write(
            f"**Movie Category:** {movie['genre']} | "
            f"**Language:** {movie['language']}"
        )

        st.write(
            movie["description"]
        )

    st.divider()


# --------------------------------------------------
# PAGE NAVIGATION
# --------------------------------------------------

previous_col, page_col, next_col = st.columns(
    [1, 2, 1]
)

with previous_col:

    if st.button(
        "⬅ Previous",
        disabled=st.session_state.movie_page == 1
    ):

        st.session_state.movie_page -= 1

        st.rerun()

with page_col:

    st.markdown(
        f"""
        <div style='text-align:center;padding-top:8px;'>
            Page {st.session_state.movie_page} of {total_pages}
        </div>
        """,
        unsafe_allow_html=True
    )

with next_col:

    if st.button(
        "Next ➡",
        disabled=(
            st.session_state.movie_page == total_pages
        )
    ):

        st.session_state.movie_page += 1

        st.rerun()