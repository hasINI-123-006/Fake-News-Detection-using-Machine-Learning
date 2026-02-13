import streamlit as st
import joblib

# Load saved model
model = joblib.load("fake_news_modelnew.pkl")

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection App")
st.write("Enter a news article below to check whether it is Real or Fake.")

# Text input
user_input = st.text_area("Enter News Text Here")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([user_input])[0]
        probability = model.predict_proba([user_input])[0]

        if prediction == 1:
            st.success("✅ This News is REAL")
        else:
            st.error("🚨 This News is FAKE")

        st.write("### Confidence Scores:")
        st.write(f"Fake: {probability[0]:.2f}")
        st.write(f"Real: {probability[1]:.2f}")

        st.progress(float(max(probability)))
