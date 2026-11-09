import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load selected dataset
df = pd.read_csv("selected_movies.csv")

# Remove rows with empty content
df = df.dropna(subset=["content", "genre", "title"])

# Split dataset into training and testing data
train_data, test_data = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)

print("Training movies:", len(train_data))
print("Testing movies:", len(test_data))


def evaluate_model(vectorizer, model_name):

    # Learn vocabulary from training data
    train_matrix = vectorizer.fit_transform(train_data["content"])

    # Transform testing data
    test_matrix = vectorizer.transform(test_data["content"])

    total_similarity = 0
    total_genre_matches = 0
    total_recommendations = 0
    recommended_movies = set()

    for i in range(test_matrix.shape[0]):

        similarities = cosine_similarity(
            test_matrix[i],
            train_matrix
        ).flatten()

        top_indices = similarities.argsort()[-5:][::-1]

        for index in top_indices:

            recommended_movies.add(index)

            if (
                str(test_data.iloc[i]["genre"]).strip().lower()
                ==
                str(train_data.iloc[index]["genre"]).strip().lower()
            ):
                total_genre_matches += 1

            total_similarity += similarities[index]
            total_recommendations += 1

    average_similarity = (
        total_similarity / total_recommendations
    )

    genre_match_rate = (
        total_genre_matches / total_recommendations
    ) * 100

    coverage = (
        len(recommended_movies) / len(train_data)
    ) * 100

    return {
        "Algorithm": model_name,
        "Average Similarity": round(average_similarity, 4),
        "Genre Match (%)": round(genre_match_rate, 2),
        "Recommendation Coverage (%)": round(coverage, 2)
    }


# Compare both algorithms
results = []

results.append(
    evaluate_model(
        CountVectorizer(stop_words="english"),
        "CountVectorizer"
    )
)

results.append(
    evaluate_model(
        TfidfVectorizer(stop_words="english"),
        "TF-IDF"
    )
)

# Display results
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON RESULTS")
print(results_df)

# Save results
results_df.to_csv("model_comparison_results.csv", index=False)

# Create comparison graph
results_df.set_index("Algorithm").plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("ML Algorithm Comparison")
plt.ylabel("Score (%) or Similarity")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("model_comparison_graph.png")
plt.close()

print("\nComparison completed successfully!")
print("Results saved in model_comparison_results.csv")
print("Graph saved as model_comparison_graph.png")