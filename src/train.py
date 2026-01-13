from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd

data = [
    ("ham",  "Are we meeting at 6 today?"),
    ("spam", "Win a free iPhone now!!! Click here"),
    ("ham",  "Please call me when you are free"),
    ("spam", "Congratulations! You won a prize. Claim now"),
    ("ham",  "I am coming home in 10 minutes"),
    ("spam", "Urgent! Your account is blocked. Verify now"),
    ("ham",  "Can you send the report today?"),
    ("spam", "Get cheap loans. Limited offer"),
]

labels = [label[0] for label in data]
messages = [message[1] for message in data]
print(labels)
print(messages)
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(messages)
# print(x)
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(x_train, y_train)

pred = model.predict(x_test)
print(classification_report(y_test, pred))

# Ensure the directory exists
Path("models").mkdir(parents=True, exist_ok=True)

# Save models using relative paths which are safer and cleaner
joblib.dump(model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/tfidf.pkl")
print("Saved model files to models/")
