import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("news.csv")

# Features and Labels
X = data["text"]
y = data["label"]

# -----------------------------
# Convert Text into Numerical Form
# -----------------------------
vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# -----------------------------
# Train Machine Learning Model
# -----------------------------
model = LogisticRegression()

model.fit(X_vectorized, y)

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("📰 Fake News Detector")

st.write(
    "Enter a news article and the AI model will predict whether it is REAL or FAKE."
)

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area(
    "Enter News Content"
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Check News"):

    if user_input != "":

        # Convert Input Text
        input_data = vectorizer.transform([user_input])

        # Prediction
        prediction = model.predict(input_data)

        # Prediction Probability
        probability = model.predict_proba(input_data)

        confidence = round(max(probability[0]) * 100, 2)

        # -----------------------------
        # Show Result
        # -----------------------------
        st.subheader("Prediction Result")

        if prediction[0] == "FAKE":

            st.error("❌ This News is FAKE")

        else:

            st.success("✅ This News is REAL")

        # -----------------------------
        # Confidence Score
        # -----------------------------
        st.write(f"### Confidence Score: {confidence}%")

    else:

        st.warning("Please enter news content.")
