# 🎬 Movie Recommendation System

A Machine Learning-based Movie Recommendation System developed using Python, Scikit-learn, Pandas, and Streamlit.

This project uses **Content-Based Filtering** to recommend movies based on their genres and descriptions. It calculates similarity between movies using CountVectorizer and Cosine Similarity.

The application provides an interactive interface where users can explore movies, apply genre and language filters, receive recommendations, create accounts, log in, and manage their favourite movies.

## 1. Aim

The main aim of this project is to design and develop a Machine Learning-based Movie Recommendation System that suggests similar movies based on their content.

The system analyzes movie genres and descriptions, converts textual information into numerical representations, and calculates similarity between movies using Cosine Similarity.

## 2. Objectives

- To develop a movie recommendation system using Machine Learning.
- To implement Content-Based Filtering.
- To collect and organize movie information in a structured dataset.
- To perform data preprocessing and cleaning.
- To select relevant features such as genre and description.
- To convert textual information into numerical vectors.
- To calculate similarity using Cosine Similarity.
- To compare CountVectorizer and TF-IDF techniques.
- To perform Exploratory Data Analysis (EDA).
- To experiment with different model configurations.
- To develop an interactive Streamlit application.
- To implement genre and language filtering.
- To provide user registration and login.
- To implement favourite movie management.

## 3. Problem Statement

A large number of movies are available across different genres and languages. Users often face difficulty selecting movies because they may not know which movies are similar to their interests.

Searching manually through a movie collection can be time-consuming.

To address this problem, this project develops a Machine Learning-based Movie Recommendation System that recommends movies according to their content.

The system analyzes movie genres and descriptions, converts textual information into numerical vectors using CountVectorizer, and measures similarity using Cosine Similarity.

### Input
- A movie selected by the user.
- Movie genre and description information.

### Output
- A list of recommended movies.
- Similarity scores.
- Movie titles and posters.

### Machine Learning Task
Content-Based Recommendation using text feature extraction and similarity measurement.

## 4. Dataset Information

The project uses a custom-curated movie dataset containing **48 movie records**.

### Dataset Attributes

| Attribute | Description |
|---|---|
| movie_id | Unique identification number |
| title | Name of the movie |
| genre | Movie category |
| description | Short movie description |
| poster | Movie poster reference |
| language | Language of the movie |

### Dataset Details

- Total movie records: 48
- Total attributes: 6
- Languages: English, Hindi, and Marathi
- File format: CSV
- Main dataset: `movies.csv`

### Feature Usage

The recommendation model uses:

1. Genre
2. Description

These two features are combined to create the textual content used by the recommendation algorithm.

Other attributes are used by the application:

- Title: Movie identification and selection.
- Language: Language-based filtering.
- Poster: Displaying movie images.
- Movie ID: Movie identification.

Language, poster, and movie ID are not included in the recommendation model's text vectorization.

## 5. Machine Learning Pipeline

The following pipeline describes the development process.

### Step 1: Data Collection

A custom-curated dataset containing 48 movie records was prepared in CSV format.

### Step 2: Data Preprocessing

The dataset was prepared for analysis and model development.

Operations include:

- Checking missing values.
- Removing duplicate records.
- Handling missing text values.
- Removing unnecessary whitespace.
- Preparing textual data for feature extraction.

### Step 3: Exploratory Data Analysis

EDA was performed to understand the dataset.

Visualizations include:

- Genre distribution.
- Language distribution.
- Description length distribution.

Matplotlib was used for graphical visualization.

### Step 4: Feature Selection

The relevant features selected for recommendation are:

- Genre
- Description

Both features are combined into a single content feature.

### Step 5: Feature Extraction

CountVectorizer converts textual movie information into numerical vectors.

TF-IDF Vectorizer was also implemented for comparison.

### Step 6: Train-Test Split

The dataset was divided into training and testing subsets using an 80:20 ratio.

| Dataset | Records |
|---|---:|
| Total records | 48 |
| Training records | 38 |
| Testing records | 10 |
| Split ratio | 80:20 |
| Random state | 42 |

This split is used for the notebook experiment. The final content-based recommendation model is fitted using the complete movie collection.

### Step 7: Model Comparison

CountVectorizer and TF-IDF were studied using project-defined recommendation diagnostic measures.

### Step 8: Similarity Calculation

Cosine Similarity calculates the similarity between movie-content vectors.

### Step 9: Model Improvement

Different CountVectorizer configurations were experimented with to observe their effects on:

- Average similarity.
- Genre match.
- Recommendation coverage.

### Step 10: Final Model

The final notebook model uses CountVectorizer with the complete 48-movie dataset.

The Streamlit application independently performs the same content-based recommendation process when it runs.

### Step 11: Recommendation Output

Movies are ranked according to their similarity scores, and the application displays the recommended movies.

### ML Pipeline Diagram

```text
             Movie Dataset
                   |
                   v
           Data Preprocessing
                   |
                   v
        Exploratory Data Analysis
                   |
                   v
            Feature Selection
                   |
                   v
          Genre + Description
                   |
                   v
            CountVectorizer
                   |
                   v
          Numerical Vectors
                   |
                   v
           Cosine Similarity
                   |
                   v
          Similarity Ranking
                   |
                   v
          Recommended Movies
                   |
                   v
           Streamlit Frontend
```

### System Architecture

```text
                 USER
                   |
                   v
          STREAMLIT APPLICATION
                   |
          +--------+---------+
          |                  |
          v                  v
     MOVIE DATASET      SQLITE DATABASE
          |                  |
          v                  v
    CONTENT FEATURES    USER ACCOUNTS
          |             AND FAVOURITES
          v
     COUNTVECTORIZER
          |
          v
    NUMERICAL VECTORS
          |
          v
    COSINE SIMILARITY
          |
          v
   RECOMMENDATION RESULTS
          |
          v
     DISPLAY MOVIE CARDS
```

### Feature Scaling / Normalization

The selected features are textual data rather than independent numerical measurement features. Therefore, a separate numerical scaling technique such as StandardScaler is not required.

Cosine Similarity is used to compare the direction of the numerical text vectors.

## 6. Algorithms and Techniques Used

### A. CountVectorizer

CountVectorizer is a text feature extraction technique provided by Scikit-learn.

It converts textual information into numerical vectors based on word frequencies.

In this project:

- Genre and description are combined.
- Text is converted into numerical vectors.
- Vectors are used to calculate movie similarity.

Advantages:

- Simple to implement.
- Suitable for text feature extraction.
- Converts textual information into numerical representations.

### B. TF-IDF Vectorizer

TF-IDF stands for Term Frequency-Inverse Document Frequency.

It assigns importance to words based on their frequency in a document and across the document collection.

It was implemented as an alternative feature extraction technique for comparison.

### C. Cosine Similarity

Cosine Similarity measures the similarity between two numerical vectors.

It calculates the cosine of the angle between the vectors.

Role in the project:

- Compares movie-content vectors.
- Calculates similarity scores.
- Identifies movies with similar content.
- Supports recommendation ranking.

## 7. Experimental Results

Experiments were conducted to study different text vectorization techniques and CountVectorizer configurations.

### A. Train-Test Split

| Parameter | Value |
|---|---:|
| Total records | 48 |
| Training records | 38 |
| Testing records | 10 |
| Split ratio | 80:20 |
| Random state | 42 |

### B. Model Comparison

CountVectorizer and TF-IDF were implemented and compared in the project.

The notebook includes the feature extraction comparison. The final recommendation implementation uses CountVectorizer.

### C. Model Improvement Experiments

Three CountVectorizer configurations were tested using the updated 48-movie dataset.

| Experiment | Average Similarity | Genre Match (%) | Coverage (%) |
|---|---:|---:|---:|
| Default CountVectorizer | 0.0402 | 54.17 | 97.92 |
| Unigrams + Bigrams | 0.0248 | 54.58 | 97.92 |
| Minimum Document Frequency 2 | 0.0707 | 57.08 | 100.00 |

### Result Analysis

- Default CountVectorizer achieved an average similarity of 0.0402.
- Unigrams and bigrams achieved a genre match of 54.58%.
- Minimum Document Frequency 2 achieved the highest average similarity and genre match.
- Minimum Document Frequency 2 also achieved 100% coverage.
- Default CountVectorizer was retained for the final implementation to maintain consistency with the existing application.

**Evaluation Note:** These are project-specific diagnostic measures, not conventional classification accuracy. Average similarity, genre match, and coverage measure different aspects of recommendation behaviour. They should not be interpreted as proof of user satisfaction or recommendation accuracy.

## 8. Final Recommendation Model

The final recommendation approach uses Content-Based Filtering.

### Final Model Configuration

| Component | Selected Method |
|---|---|
| Recommendation technique | Content-Based Filtering |
| Content features | Genre + Description |
| Feature extraction | CountVectorizer |
| Similarity measurement | Cosine Similarity |
| Dataset size | 48 records |
| Recommendation ranking | Similarity scores |

### Final Model Process

1. Movie genre and description are combined.
2. CountVectorizer converts textual content into numerical vectors.
3. Cosine Similarity calculates similarity between movie vectors.
4. Similarity scores are ranked.
5. The system returns similar movies.

### Model Integration

The Jupyter Notebook contains the ML development and experimentation process.

The Streamlit application independently trains its recommendation model when the application runs.

The current `app.py` implementation does not load `movie_recommender_model.joblib`; it creates the CountVectorizer feature matrix and Cosine Similarity matrix dynamically from `movies.csv`.

## 9. Frontend and Application Features

The frontend is developed using Streamlit.

### 1. User Registration

New users can create an account.

### 2. User Login

Registered users can log in to the application.

### 3. User Profile

The application displays the logged-in user's profile information.

### 4. Discover Movies

Users can explore the available movie collection.

### 5. Genre Filtering

Users can filter movies according to genre.

### 6. Language Filtering

Users can filter movies based on English, Hindi, and Marathi languages.

### 7. Movie Recommendations

Users can select a movie and receive similar movie suggestions.

### 8. Movie Posters

Movie posters are displayed along with movie information when available.

### 9. My Favorites

Users can add movies to their favourite list.

### 10. Remove Favorites

Users can remove movies from their favourite list.

### Database

SQLite is used to manage user accounts and favourite movies.

Database operations are implemented separately in `database.py`.

## 10. Screenshots

The following screenshots demonstrate the working application and analysis.

### A. Movie Application
Main application interface with navigation, filters, and user profile.

### B. Movie List
Available movies displayed with their titles and posters.

### C. Movie Recommendations
Recommendation results generated after selecting a movie.

### D. Genre Distribution
Graphical representation of movie genres.

### E. Language Distribution
Graphical representation of movie languages.

### F. Description Length Distribution
Visualization of movie description lengths.

*Add the corresponding screenshots and EDA graph images from the Screenshots and eda_graphs folders when displaying this README.*

## 11. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Dataset handling |
| Scikit-learn | Machine Learning and text vectorization |
| CountVectorizer | Text feature extraction |
| TF-IDF | Alternative text feature extraction |
| Cosine Similarity | Similarity calculation |
| Matplotlib | Data visualization |
| Streamlit | Frontend application |
| SQLite | User and favourites database |
| Joblib | Model serialization support |
| Jupyter Notebook | ML development and experimentation |
| Git and GitHub | Version control and project hosting |

## 12. Installation and Execution

### Prerequisites

- Python 3.x
- Visual Studio Code (recommended)
- Internet connection for installing dependencies

### Step 1: Clone the Repository

```bash
git clone https://github.com/ranikajale706-art/Movie-Recommendation-System.git
```

### Step 2: Navigate to the Project Folder

```bash
cd Movie-Recommendation-System
```

### Step 3: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
streamlit run app.py
```

### Step 5: Open the Application

Streamlit provides a local address in the terminal.

Open that address in a web browser to access the Movie Recommendation System.

## 13. GitHub Repository

**Repository Name:** Movie Recommendation System

**GitHub:** https://github.com/ranikajale706-art/Movie-Recommendation-System

The repository contains the project source code, dataset, ML notebook, documentation, screenshots, and supporting files.

## 14. Project Summary

The Movie Recommendation System demonstrates the practical application of Machine Learning and Natural Language Processing concepts in movie discovery.

The project uses a custom-curated dataset containing 48 movie records. Data preprocessing, feature selection, and Exploratory Data Analysis are performed to prepare the movie information for recommendation.

CountVectorizer and TF-IDF are implemented for text feature extraction, while Cosine Similarity is used to measure similarity between movies.

Different CountVectorizer configurations are also experimented with to study recommendation similarity, genre match, and coverage.

The Streamlit application provides movie discovery, genre and language filtering, movie recommendations, user registration, login, and favourite movie management.

SQLite is used to store user account and favourite movie information.

The project demonstrates how content-based Machine Learning techniques can be integrated into an interactive movie recommendation application.