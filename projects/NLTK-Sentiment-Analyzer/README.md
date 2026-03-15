# NLTK Sentiment Analyzer

This project is a simple but effective sentiment analysis tool built using Python and the Natural Language Toolkit (NLTK). It's trained on a dataset of movie reviews to classify a piece of text as either "Positive" or "Negative".

## Project Overview

The script performs the following steps:
1.  **Loads Data:** It uses the `movie_reviews` corpus included with NLTK.
2.  **Text Preprocessing:** Cleans text by tokenizing, converting to lowercase, and removing stopwords and punctuation.
3.  **Feature Extraction:** Uses a "Bag of Words" approach based on the 3,000 most frequent words.
4.  **Model Training:** A Naive Bayes classifier is trained on 95% of the dataset.
5.  **Evaluation:** The model is evaluated on the remaining data to check its accuracy.
6.  **Prediction:** A function is provided to predict the sentiment of any new sentence.

## Technologies Used
- Python 3
- NLTK (Natural Language Toolkit)

## How to Run

1.  **Navigate to the project directory:**
    From the root of the repository, run the following in the terminal:
    ```bash
    cd NLTK-Sentiment-Analyzer
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the script:**
    ```bash
    python sentiment_analyzer.py
    ```
    The first time you run the script, it will automatically download the necessary NLTK data packages.

## Sample Output