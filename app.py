# app.py
import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Paths to model and vectorizer in project root
# -----------------------------
MODEL_PATH = "model.pkl"
VECT_PATH = "vectorizer.pkl"

@st.cache_resource
def load_objects():
    # Load the trained RandomForest model and TF-IDF vectorizer using pickle
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECT_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")
st.title("📩 SMS Spam Detector")
st.write("Enter an SMS message and the model will predict whether it's **spam** or **ham** (not spam).")

# Initialize session state
if "message" not in st.session_state:
    st.session_state["message"] = ""

model, vectorizer = load_objects()

# -----------------------------
# Example messages
# -----------------------------
examples = [
    "Free entry in a weekly competition to win FA Cup final tickets. Text WIN to 12345",
    "Hey, are you coming to class tomorrow?",
    "Claim your free voucher now! Reply YES to get it."
]

st.subheader("Try an example")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Example 1"):
        st.session_state["message"] = examples[0]

with col2:
    if st.button("Example 2"):
        st.session_state["message"] = examples[1]

with col3:
    if st.button("Example 3"):
        st.session_state["message"] = examples[2]

# -----------------------------
# Message input
# -----------------------------
message = st.text_area("SMS message", value=st.session_state["message"], height=150)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    if not message.strip():
        st.warning("Please enter a message to classify.")
    else:
        vect = vectorizer.transform([message])
        X = vect.toarray()  # RandomForest requires dense input
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]  # probability of spam

        label = "📛 Spam" if pred == 1 else "✅ Ham (Not Spam)"
        st.write("### Prediction Result")
        st.success(f"**{label}** — probability(spam) = {proba:.3f}")

        # Optional: show top features for RandomForest
        try:
            feat_names = vectorizer.get_feature_names_out()
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[-10:][::-1]
            top_words = [(feat_names[i], float(importances[i])) for i in top_idx if importances[i] > 0]

            if top_words:
                st.write("### 🔍 Important Features (approx):")
                st.table(top_words)
        except Exception:
            pass

st.markdown("---")
st.caption("Model: RandomForestClassifier using TF-IDF features.")
