import pandas as pd
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load selected dataset
df = pd.read_csv("selected_movies.csv")
df = df.dropna(subset=["content", "title"]).reset_index(drop=True)

# Create final ML model
vectorizer = CountVectorizer(stop_words="english")

# Train model using movie content
movie_matrix = vectorizer.fit_transform(df["content"])

# Calculate movie similarity
similarity_matrix = cosine_similarity(movie_matrix)

# Save trained model and required data
model_data = {
    "vectorizer": vectorizer,
    "similarity_matrix": similarity_matrix,
    "titles": df["title"].tolist()
}

joblib.dump(model_data, "movie_recommender_model.joblib")

print("Final model trained successfully!")
print("Total movies:", len(df))
print("Model saved as movie_recommender_model.joblib")
