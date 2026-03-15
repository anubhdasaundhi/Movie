import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load cleaned dataset
df = pd.read_csv("cleaned_data.txt", sep="\t")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.split()
    return " ".join(tokens)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
tfidf_matrix = vectorizer.fit_transform(df["cleaned_storyline"])

# Recommendation function
def recommend_movies(user_input):

    cleaned_input = clean_text(user_input)

    input_vector = vectorizer.transform([cleaned_input])

    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)

    top_indices = similarity_scores.argsort()[0][-5:][::-1]

    return df.iloc[top_indices][["Movie Name", "Storyline"]]


# STREAMLIT UI
st.title("🎬 IMDB Movie Recommendation System")

st.write("Enter a movie storyline and get similar movie recommendations.")

user_input = st.text_area("Enter Storyline")

if st.button("Recommend Movies"):

    if user_input.strip() == "":
        st.warning("Please enter a storyline.")
    else:
        results = recommend_movies(user_input)

        st.subheader("Top Recommended Movies")

        for index, row in results.iterrows():
            st.write("### 🎥", row["Movie Name"])
            st.write(row["Storyline"])
            st.write("---")