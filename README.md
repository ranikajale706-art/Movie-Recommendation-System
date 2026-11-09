# Movie Recommendation System

## 1. Project Overview
The Movie Recommendation System is a Machine Learning-based application that recommends movies based on their genres and descriptions.

The application helps users discover movies according to their interests, preferred categories and languages.

## 2. Problem Statement
Users often find it difficult to select movies from a large collection. This project provides a content-based recommendation system that helps users discover similar movies.

## 3. Aim
To develop a movie recommendation application using Machine Learning techniques.

## 4. Objectives
- Recommend similar movies.
- Provide genre-based movie exploration.
- Support multiple languages.
- Allow user registration and login.
- Allow users to save favorite movies.

## 5. Dataset Information
The project uses a movie dataset containing 47 records.

Dataset features:
- Movie ID
- Movie Title
- Genre
- Description
- Poster
- Language

## 6. Data Preprocessing
The following preprocessing techniques were applied:
- Duplicate records removal.
- Missing text value handling.
- Whitespace removal.
- Text correction.
- Dataset cleaning.

## 7. Feature Selection
The following features were selected:
- Genre
- Description
- Language

Genre and description were combined to create the movie content feature.

## 8. Exploratory Data Analysis
EDA was performed using Python, Pandas and Matplotlib.

Visualizations:
- Genre distribution.
- Language distribution.
- Movie description length distribution.

## 9. Machine Learning Algorithms

Two content-based approaches were compared:

1. CountVectorizer with Cosine Similarity.
2. TF-IDF with Cosine Similarity.

## 10. Model Comparison

| Metric | CountVectorizer | TF-IDF |
|---|---:|---:|
| Average Similarity | 0.3453 | 0.2923 |
| Genre Match | 56% | 52% |
| Coverage | 78.38% | 78.38% |

These are proxy evaluation measures, not conventional recommendation accuracy.

## 11. Model Improvement
Three parameter configurations were tested.

| Experiment | Average Similarity | Genre Match | Coverage |
|---|---:|---:|---:|
| Default | 0.3453 | 56% | 78.38% |
| Unigrams + Bigrams | 0.2679 | 56% | 78.38% |
| Minimum Document Frequency 2 | 0.5411 | 58% | 67.57% |

The default configuration was selected for its higher recommendation coverage.

## 12. Final Model
CountVectorizer and Cosine Similarity were selected for the final recommendation model.

The trained model is saved as:

`movie_recommender_model.joblib`

## 13. Frontend
The application is developed using Streamlit.

Features:
- User registration and login.
- Movie recommendations.
- Genre filtering.
- Language filtering.
- Movie favorites.

## 14. Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- SQLite
- GitHub

## 15. How to Run the Project

Install the required libraries:

`python -m pip install -r requirements.txt`

Run the application:

`streamlit run app.py`

## 16. Project Repository
GitHub:
https://github.com/ranikajale706-art/Movie-Recommendation-System

## 17. Conclusion
The project demonstrates a content-based movie recommendation approach using text vectorization and cosine similarity. It provides users with a simple interface to explore movies and receive recommendations.