import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("cleaned_movies.csv")

# Create graph folder
Path("eda_graphs").mkdir(exist_ok=True)

print("Dataset Shape:", df.shape)

# Graph 1: Genre Distribution
plt.figure(figsize=(10, 5))
df["genre"].value_counts().plot(kind="bar")
plt.title("Movies by Genre")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_graphs/genre_distribution.png")
plt.close()

# Graph 2: Language Distribution
plt.figure(figsize=(7, 5))
df["language"].value_counts().plot(kind="bar")
plt.title("Movies by Language")
plt.xlabel("Language")
plt.ylabel("Number of Movies")
plt.tight_layout()
plt.savefig("eda_graphs/language_distribution.png")
plt.close()

# Graph 3: Description Length
df["description_length"] = (
    df["description"].fillna("").str.split().str.len()
)

plt.figure(figsize=(8, 5))
plt.hist(df["description_length"], bins=8)
plt.title("Description Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Movies")
plt.tight_layout()
plt.savefig("eda_graphs/description_distribution.png")
plt.close()

print("\nEDA completed successfully!")
print("All three graphs saved in eda_graphs folder.")