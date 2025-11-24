import streamlit as st
import pickle
import string
import re

# ---------------------------
# Load Model and Vectorizer
# ---------------------------

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_vectorizer():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer

model = load_model()
vectorizer = load_vectorizer()

# ---------------------------
# Text Preprocessing Function
# ---------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = "".join([c for c in text if c not in string.punctuation])
    return text

# ---------------------------
# Streamlit App UI
# ---------------------------

st.title("📩 SMS Spam Detector")
st.write("Enter a message to check if it is **Spam** or **Ham (Not Spam)**.")

user_input = st.text_area("Type your message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a valid message!")
    else:
        processed = clean_text(user_input)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.error("🚨 **Prediction: SPAM**")
        else:
            st.success("✅ **Prediction: NOT SPAM (HAM)**")
