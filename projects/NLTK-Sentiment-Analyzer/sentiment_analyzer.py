# sentiment_analyzer.py

import nltk
import random
from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize
import string

# --- One-time Downloads for NLTK ---
# This section ensures that the necessary NLTK data is downloaded.
# It tries to find the data, and if a LookupError occurs, it downloads it.
try:
    nltk.data.find('corpora/movie_reviews')
except LookupError:
    print("Downloading 'movie_reviews' corpus...")
    nltk.download('movie_reviews')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading 'punkt' tokenizer...")
    nltk.download('punkt')

# --- NEWLY ADDED FIX ---
# The word_tokenize function has a dependency on 'punkt_tab'.
# This is not always included with the main 'punkt' download and is needed
# for the prediction part of the script. We add a check for it here.
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading 'punkt_tab' for tokenizer...")
    nltk.download('punkt_tab')
# --- END OF FIX ---

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading 'stopwords' corpus...")
    nltk.download('stopwords')

# --- 1. Data Preparation ---

print("Preparing data...")

# Load stopwords and punctuation
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# This function cleans up the text by tokenizing, converting to lowercase,
# and removing stopwords and punctuation.
def preprocess_text(text):
    tokens = word_tokenize(text)
    # Filter out stopwords and punctuation, and convert to lowercase
    filtered_tokens = [
        word.lower() for word in tokens
        if word.lower() not in stop_words and word not in punctuation
    ]
    return filtered_tokens

# Load movie reviews and their categories ('pos' or 'neg')
documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        # Each document is a tuple of (list_of_words, category)
        documents.append((list(movie_reviews.words(fileid)), category))

# Shuffle the documents to ensure random distribution for training/testing
random.shuffle(documents)

print(f"Loaded and shuffled {len(documents)} movie reviews.")

# --- 2. Feature Extraction ---

# Get all words from all reviews to form our feature vocabulary
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

# Use NLTK's Frequency Distribution to find the most common words
all_words_freq = nltk.FreqDist(all_words)

# We will use the 3000 most common words as our features
word_features = list(all_words_freq.keys())[:3000]

# This function converts a list of words (a document) into a feature set.
# The feature set is a dictionary where keys are the top 3000 words and
# values are True/False, indicating if the word is present in the document.
# This is known as a "Bag of Words" model.
def find_features(document_words):
    words = set(document_words)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

# Create feature sets for all our documents
featuresets = [(find_features(rev), category) for (rev, category) in documents]
print("Feature sets created.")

# --- 3. Training and Testing the Classifier ---

# Split the data into a training set (first 1900 reviews) and a testing set (last 100 reviews)
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

print("Training the classifier...")
# Train the Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier trained successfully.")

# --- 4. Evaluating the Classifier ---

# Test the accuracy of our classifier on the testing set
accuracy = nltk.classify.accuracy(classifier, testing_set) * 100
print(f"Classifier Accuracy: {accuracy:.2f}%")

# Show the most informative features the classifier found
print("\nMost Informative Features:")
classifier.show_most_informative_features(15)

# --- 5. Using the Classifier for Prediction ---

def predict_sentiment(text):
    """
    Predicts the sentiment of a given text string.
    """
    words = preprocess_text(text)
    feats = find_features(words)
    return classifier.classify(feats)

# --- Main execution block to demonstrate the predictor ---
if __name__ == "__main__":
    print("\n--- Sentiment Prediction Demo ---")
    
    custom_review_1 = "The movie was absolutely brilliant! The acting was superb and the plot was gripping."
    sentiment_1 = predict_sentiment(custom_review_1)
    print(f"Review: '{custom_review_1}'")
    print(f"Predicted Sentiment: {'Positive' if sentiment_1 == 'pos' else 'Negative'}\n")

    custom_review_2 = "It was a complete waste of time. The plot was predictable and the acting was terrible."
    sentiment_2 = predict_sentiment(custom_review_2)
    print(f"Review: '{custom_review_2}'")
    print(f"Predicted Sentiment: {'Positive' if sentiment_2 == 'pos' else 'Negative'}\n")

    custom_review_3 = "The film was okay, not great but not bad either. It had some good moments."
    sentiment_3 = predict_sentiment(custom_review_3)
    print(f"Review: '{custom_review_3}'")
    print(f"Predicted Sentiment: {'Positive' if sentiment_3 == 'pos' else 'Negative'}\n")