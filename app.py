#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import requests


# URLs of the model and vectorizer
model_url = 'https://raw.githubusercontent.com/matthewtech-o/coding_assessment/main/classifier_model.pkl'
vectorizer_url = 'https://raw.githubusercontent.com/matthewtech-o/coding_assessment/main/tfidf_vectorizer.pkl'

# Local paths to save the downloaded files
model_path = '/tmp/classifier_model.pkl'
vectorizer_path = '/tmp/vectorizer.pkl'

# Function to download a file from a URL and save it locally
def download_file(url, local_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download file from {url}")

# Download the model and vectorizer
download_file(model_url, model_path)
download_file(vectorizer_url, vectorizer_path)

# Load your pre-trained model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

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
