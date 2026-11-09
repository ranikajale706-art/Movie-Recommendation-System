import joblib

# Load saved model
model = joblib.load("movie_recommender_model.joblib")

titles = model["titles"]
similarity = model["similarity_matrix"]

# Test movie
movie_name = "Interstellar"

if movie_name in titles:

    index = titles.index(movie_name)

    scores = list(enumerate(similarity[index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    print("Recommendations for:", movie_name)

    count = 0

    for movie_index, score in scores:

        if movie_index != index:
            print(titles[movie_index])
            count += 1

        if count == 5:
            break

else:
    print("Movie not found!")