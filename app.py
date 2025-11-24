# app.py
import streamlit as st
import joblib
import numpy as np
import os

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
VECT_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")

@st.cache_resource
def load_objects():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    return model, vectorizer

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")
st.title("📩 SMS Spam Detector")
st.write("Enter an SMS message and the model will predict whether it's **spam** or **ham** (not spam).")

model, vectorizer = load_objects()

# Example messages
examples = [
    "Free entry in 2 a weekly competition to win FA Cup final tickets. Text WIN to 12345",
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

# Message input
message = st.text_area("SMS message", value=st.session_state.get("message", ""), height=150)
if st.button("Predict"):
    if not message.strip():
        st.warning("Please enter a message to classify.")
    else:
        vect = vectorizer.transform([message])
        # convert to dense for RandomForest (if you trained with dense)
        X = vect.toarray()
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]  # probability of spam
        label = "Spam" if pred == 1 else "Ham (Not spam)"
        st.write("### Prediction")
        st.success(f"**{label}** — probability(spam) = {proba:.3f}")

        # Show some explanation: top features (simple)
        try:
            import numpy as np
            feat_names = vectorizer.get_feature_names_out()
            # Use model.feature_importances_ to show top contributing words
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[-10:][::-1]
            top = [(feat_names[i], importances[i]) for i in top_idx if importances[i] > 0]
            if top:
                st.write("Top important features (approx):")
                st.table(top)
        except Exception:
            pass

st.markdown("---")
st.write("Model: RandomForestClassifier (TF-IDF features).")
