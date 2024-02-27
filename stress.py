import pandas as pd 
import sklearn 
import streamlit as st
import joblib 
st.title("MENTAL HEALTH ISSUES")
 
st.header(":red[Enter down you mental health issue]")

Post = st.text_input("Enter Your Issue ")
 
if st.button("Predict"):
    cv = joblib.load("vectorizer.h5")
    sentence = cv.transform([Post])
    model = joblib.load("stress_model.h5")
    Prediction = model.predict(sentence)
    st.success(Prediction[0])



