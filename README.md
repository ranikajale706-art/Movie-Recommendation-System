# Movie Recommendation System

A Machine Learning-based Movie Recommendation System developed using Python and Streamlit.

## Features

- User Registration and Login
- User Profile
- Search Movies by Name
- Filter Movies by Category
- Filter Movies by Language
- Content-Based Movie Recommendations
- Movie Posters and Descriptions
- Add and Remove Favourite Movies

## Machine Learning

This project uses Content-Based Filtering.

- CountVectorizer converts movie genres and descriptions into numerical features.
- Cosine Similarity measures similarity between movies.
- The system recommends movies based on their content.

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- SQLite
- Requests
- Pillow

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
streamlit run app.py
```

## Project Files

- `app.py` – Main application
- `database.py` – Database operations
- `movies.csv` – Movie dataset
- `requirements.txt` – Required Python libraries
- `.gitignore` – Files excluded from GitHub