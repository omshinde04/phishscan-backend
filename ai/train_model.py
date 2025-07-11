import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# ✅ Sample or real data
use_sample = True  # Set to False to load CSV

if use_sample:
    data = {
        "text": [
            "Verify your account now",
            "Reset your password immediately",
            "Win a free iPhone",
            "Update your billing info",
            "Hey friend, just checking in",
            "Meeting rescheduled to Friday",
            "Lunch tomorrow?",
            "Here's the invoice you asked for"
        ],
        "label": [1, 1, 1, 1, 0, 0, 0, 0]  # 1 = phishing, 0 = safe
    }
    df = pd.DataFrame(data)
else:
    df = pd.read_csv("emails.csv")  # Make sure this file has 'text' and 'label' columns

# 🚀 Train/Test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.25, random_state=42)

# 🔧 Vectorizer + Model pipeline
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 🧠 Evaluate
preds = model.predict(X_test_vec)
acc = accuracy_score(y_test, preds)
print(f"✅ Model trained with accuracy: {acc * 100:.2f}%")

# 💾 Save
os.makedirs("ai", exist_ok=True)
joblib.dump(model, "ai/phishing_model.pkl")
joblib.dump(vectorizer, "ai/vectorizer.pkl")
print("✅ Model and vectorizer saved in ./ai/")
