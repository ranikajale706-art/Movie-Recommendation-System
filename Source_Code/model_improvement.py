import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("selected_movies.csv")
df = df.dropna(subset=["content", "genre", "title"])

# Split dataset
train_data, test_data = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)

# Different parameter settings
experiments = [
    {
        "Name": "Default",
        "ngram_range": (1, 1),
        "min_df": 1
    },
    {
        "Name": "Unigrams + Bigrams",
        "ngram_range": (1, 2),
        "min_df": 1
    },
    {
        "Name": "Minimum Document Frequency 2",
        "ngram_range": (1, 1),
        "min_df": 2
    }
]

results = []

for experiment in experiments:

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=experiment["ngram_range"],
        min_df=experiment["min_df"]
    )

    train_matrix = vectorizer.fit_transform(
        train_data["content"]
    )

    test_matrix = vectorizer.transform(
        test_data["content"]
    )

    total_similarity = 0
    genre_matches = 0
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
                genre_matches += 1

            total_similarity += similarities[index]
            total_recommendations += 1

    average_similarity = (
        total_similarity / total_recommendations
    )

    genre_match = (
        genre_matches / total_recommendations
    ) * 100

    coverage = (
        len(recommended_movies) / len(train_data)
    ) * 100

    results.append({
        "Experiment": experiment["Name"],
        "Average Similarity": round(average_similarity, 4),
        "Genre Match (%)": round(genre_match, 2),
        "Coverage (%)": round(coverage, 2)
    })

# Display results
results_df = pd.DataFrame(results)

print("\nMODEL IMPROVEMENT RESULTS")
print(results_df)

# Save results
results_df.to_csv(
    "model_improvement_results.csv",
    index=False
)

# Create graph
results_df.set_index("Experiment").plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Model Improvement Experiments")
plt.ylabel("Score (%) or Similarity")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("model_improvement_graph.png")
plt.close()

print("\nModel improvement experiment completed!")
print("Results saved successfully.")
print("Graph saved successfully.")