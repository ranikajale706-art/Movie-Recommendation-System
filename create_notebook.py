import nbformat as n

p = "Notebook/Movie_Recommendation_ML.ipynb"
b = n.v4.new_notebook()

b.cells = [
    n.v4.new_markdown_cell("""# Movie Recommendation System
## Machine Learning Individual Project
Third Year - Computer Science and Engineering

This notebook documents the machine learning workflow of the project."""),
    
    n.v4.new_markdown_cell("## 1. Import Libraries"),
    n.v4.new_code_cell("""import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
print("Libraries imported successfully!")"""),

    n.v4.new_markdown_cell("## 2. Dataset Loading"),
    n.v4.new_code_cell("""df = pd.read_csv("Dataset/selected_movies.csv")
print("Dataset shape:", df.shape)
display(df.head())"""),

    n.v4.new_markdown_cell("## 3. Data Information and Preprocessing"),
    n.v4.new_code_cell("""df.info()
print("\\nMissing values:")
print(df.isnull().sum())
print("\\nDuplicate rows:", df.duplicated().sum())"""),

    n.v4.new_markdown_cell("## 4. Exploratory Data Analysis"),
    n.v4.new_code_cell("""print(df["genre"].value_counts())
df["genre"].value_counts().plot(kind="bar", title="Genre Distribution")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()"""),

    n.v4.new_markdown_cell("## 5. Feature Selection"),
    n.v4.new_code_cell("""X = df["content"].fillna("")
print("Selected feature: content (genre + description)")
print("Total records:", len(X))"""),

    n.v4.new_markdown_cell("## 6. Train-Test Split"),
    n.v4.new_code_cell("""X_train, X_test = train_test_split(
    X, test_size=0.2, random_state=42
)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))"""),

    n.v4.new_markdown_cell("## 7. Algorithm Comparison"),
    n.v4.new_code_cell("""count_vectorizer = CountVectorizer(stop_words="english")
count_matrix = count_vectorizer.fit_transform(X)

tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(X)

print("CountVectorizer matrix:", count_matrix.shape)
print("TF-IDF matrix:", tfidf_matrix.shape)"""),

    n.v4.new_markdown_cell("## 8. Model Improvement Experiments"),
    n.v4.new_code_cell("""results = pd.read_csv(
    "Documentation/model_improvement_results.csv"
)
display(results)"""),

    n.v4.new_markdown_cell("## 9. Final Model and Recommendation"),
    n.v4.new_code_cell("""similarity = cosine_similarity(count_matrix)

def recommend_movie(movie_title, number=5):
    matches = df.index[
        df["title"].str.lower() == movie_title.lower()
    ].tolist()

    if not matches:
        print("Movie not found.")
        return

    index = matches[0]
    scores = sorted(
        enumerate(similarity[index]),
        key=lambda item: item[1],
        reverse=True
    )

    for i, score in scores[1:number+1]:
        print(df.iloc[i]["title"], "-", round(score, 3))

recommend_movie("Interstellar")"""),

    n.v4.new_markdown_cell("""## 10. Conclusion

This project uses content-based movie recommendation.
CountVectorizer converts movie descriptions and genres into
numerical features. Cosine similarity identifies similar movies.

The project compares CountVectorizer and TF-IDF.
Evaluation values are proxy measures because the dataset
does not contain user ratings or relevance labels.""")
]

b.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}

n.write(b, p)
print("Notebook created successfully:", p)
