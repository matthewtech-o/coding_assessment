#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib

# Load your pre-trained model
model = joblib.load('/Users/matthewoladiran/Downloads/classifier_model.pkl')
vectorizer = joblib.load('/Users/matthewoladiran/Downloads/tfidf_vectorizer.pkl') 

# Define a function to classify the input comments
def classify_comment(comment):
    transformed_comment = vectorizer.transform([comment])
    prediction = model.predict(transformed_comment)
    return prediction[0]

# Streamlit app layout
st.title("Comment Classifier")

st.write("Enter a comment below to classify it into one of the following categories: Veterinarian, Medical Doctor, Other.")

# Text input for the comment
comment = st.text_input("Enter comment")

if comment:
    # Classify the comment
    classification = classify_comment(comment)
    
    # Display the result
    st.write(f"The comment is classified as: **{classification}**")
