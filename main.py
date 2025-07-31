import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Import libraries
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load spaCy model
import en_core_web_sm
nlp = en_core_web_sm.load()

# Load dataset
df = pd.read_csv('dataset.csv')

# Rename review column
df = df.rename(columns={"Review Text": "review"})

# Drop rows with missing reviews
df = df.dropna(subset=["review"])

# Create sentiment labels
def get_sentiment(rating):
    if rating >= 5: 
        return "positive"
    elif rating <= 4:
        return "negative"
    else:
        return "neutral"

df["sentiment"] = df["Rating"].apply(get_sentiment)

# Remove neutral reviews
df = df[df["sentiment"] != "neutral"]

# Limit to first 500 and last 500 rows
df = pd.concat([df.head(500), df.tail(500)], ignore_index=True)

# Stopwords
stop_words = set(stopwords.words("english"))

# Step 1: Clean HTML tags and special characters
df["cleaned_review"] = df["review"].apply(lambda x: re.sub(r"<.*?>", "", str(x).lower()))
df["cleaned_review"] = df["cleaned_review"].apply(lambda x: re.sub(r"[^a-z\s]", "", x))

# Step 2: Tokenization and stopword removal
df["cleaned_review"] = df["cleaned_review"].apply(
    lambda x: " ".join([word for word in word_tokenize(x) if word not in stop_words])
)

# Step 3: Lemmatization using spaCy (efficient pipeline)
texts = df["cleaned_review"].tolist()
lemmatized_reviews = []
for doc in nlp.pipe(texts, batch_size=1000):
    lemmatized_reviews.append(" ".join([token.lemma_ for token in doc]))
df["cleaned_review"] = lemmatized_reviews

# Convert sentiment to binary
df["sentiment_numeric"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Train-test split
X = df["cleaned_review"]
y = df["sentiment_numeric"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model
model = LogisticRegression(solver="liblinear")
model.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = model.predict(X_test_tfidf)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Prediction Function
def predict_sentiment(text):
    text_cleaned = re.sub(r"<.*?>", "", text.lower())
    text_cleaned = re.sub(r"[^a-z\s]", "", text_cleaned)
    tokens = word_tokenize(text_cleaned)
    tokens = [word for word in tokens if word not in stop_words]
    doc = nlp(" ".join(tokens))
    lemmatized = " ".join([token.lemma_ for token in doc])
    vect = tfidf.transform([lemmatized])
    pred = model.predict(vect)[0]
    return "Positive" if pred == 1 else "Negative"

# Sample predictions
print("\nSample Predictions:")
print("1:", predict_sentiment("Love this product, would buy again!"))
print("2:", predict_sentiment("Terrible quality, not worth the money."))
print("3:", predict_sentiment("It's okay, nothing special."))
print("4:", predict_sentiment("Absolutely fantastic, exceeded my expectations!"))