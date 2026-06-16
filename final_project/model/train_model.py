import pandas as pd
import re
import string
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix,
classification_report
)

from sklearn.metrics.pairwise import cosine_similarity

print("Loading Dataset...")

data = pd.read_csv(
"../dataset/spam.csv"
)

data = data.rename(
columns={
"Category": "label",
"Message": "message"
}
)

print("\nDataset Preview")
print(data.head())

print("\nDataset Shape")
print(data.shape)

data["label"] = data["label"].map({
"ham": 0,
"spam": 1
})

stop_words = ENGLISH_STOP_WORDS

def clean_text(text):
   
    text = str(text).lower()

    text = re.sub(r"\d+", "", text)

    text = text.translate(
         str.maketrans("", "", string.punctuation)
)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

print("\nCleaning Messages...")

data["message"] = data["message"].apply(
clean_text
)

vectorizer = TfidfVectorizer(
ngram_range=(1, 2),
max_features=5000
)

X = vectorizer.fit_transform(
data["message"]
)

y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

cosine_scores = cosine_similarity(
X_test[:100],
X_train[:100]
)

average_cosine = cosine_scores.mean()

print("\nAverage Cosine Similarity:")
print(round(average_cosine, 4))

def dice_similarity(a, b):

    a = set(a.split())
    b = set(b.split())

    intersection = len(
        a.intersection(b)
    )
    denominator = len(a) + len(b)

    if denominator == 0:
     return 0

    return (
    2 * intersection
) / denominator

sample1 = data["message"].iloc[0]
sample2 = data["message"].iloc[1]

dice_score = dice_similarity(
sample1,
sample2
)

print("\nDice Similarity:")
print(round(dice_score, 4))

print("\nTraining Naive Bayes Model...")

model = MultinomialNB()

model.fit(
X_train,
y_train
)

y_pred = model.predict(
X_test
)

accuracy = accuracy_score(
y_test,
y_pred
)

precision = precision_score(
y_test,
y_pred
)

recall = recall_score(
y_test,
y_pred
)

f1 = f1_score(
y_test,
y_pred
)

print("\nRESULTS")
print("=" * 50)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix")
print(confusion_matrix(
y_test,
y_pred
))

print("\nClassification Report")
print(classification_report(
y_test,
y_pred
))

os.makedirs(
"models",
exist_ok=True
)

joblib.dump(
model,
"models/spam_model.pkl"
)

joblib.dump(
vectorizer,
"models/vectorizer.pkl"
)

print("\nModel Saved Successfully")
print("spam_model.pkl created")
print("vectorizer.pkl created")
