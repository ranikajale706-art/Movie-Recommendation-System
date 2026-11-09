import pandas as pd

# Load original dataset
df = pd.read_csv("movies.csv")

print("Original Dataset Shape:", df.shape)

# Remove duplicate records
df = df.drop_duplicates()

# Remove duplicate movie IDs
df = df.drop_duplicates(subset=["movie_id"], keep="first")

# Clean text columns
text_columns = ["title", "genre", "description", "language"]

for column in text_columns:
    df[column] = df[column].fillna("").astype(str).str.strip()

# Remove extra spaces
for column in text_columns:
    df[column] = df[column].str.replace(r"\s+", " ", regex=True)

# Correct typing error
df["description"] = df["description"].replace(
    "A dWalteretective investigates a family's puzzling case",
    "A detective investigates a family's puzzling case"
)

# Save cleaned dataset
df.to_csv("cleaned_movies.csv", index=False)

print("\nPreprocessing completed!")
print("Cleaned Dataset Shape:", df.shape)
print("\nCleaned Dataset:")
print(df[["movie_id", "title", "genre", "language"]])

print("\nCleaned dataset saved successfully!")