import pandas as pd

# Load cleaned dataset
df = pd.read_csv("cleaned_movies.csv")

# Select important features
selected_features = df[
    ["movie_id", "title", "genre", "description", "language"]
].copy()

# Combine genre and description
selected_features["content"] = (
    selected_features["genre"].fillna("") + " " +
    selected_features["description"].fillna("")
)

# Save selected features
selected_features.to_csv("selected_movies.csv", index=False)

print("Feature selection completed!")
print("\nSelected Features:")
print(selected_features.head())

print("\nSelected dataset shape:", selected_features.shape)
print("\nFile saved as selected_movies.csv")