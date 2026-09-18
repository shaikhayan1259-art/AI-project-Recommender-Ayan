import streamlit as st
import pandas as pd
from recommendation_engine import ProjectRecommender

st.set_page_config(page_title="Project Topic Recommender", page_icon="🎯", layout="centered")

st.title("🎯 Project Topic Recommender")
st.caption("Profile-based project recommendations using TF-IDF + cosine similarity.")

@st.cache_resource
def load_engine():
    return ProjectRecommender("projects.csv")

engine = load_engine()

st.subheader("Student Profile")
interests = st.text_input("Interests", "artificial intelligence, data science, automation")
skills = st.text_input("Skills", "python, pandas, machine learning")
career_goal = st.text_input("Career Goal", "AI engineer")

top_n = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Recommend Projects", type="primary"):
    profile = f"{interests} {skills} {career_goal}"
    results = engine.recommend(profile, top_n=top_n)

    st.subheader("Recommended Projects")
    for _, row in results.iterrows():
        st.markdown(f"### {row['title']}")
        st.write(row["description"])
        st.write(f"**Domain:** {row['domain']}  |  **Difficulty:** {row['difficulty']}")
        st.write(f"**Skills:** {row['skills']}")
        st.write(f"**Match score:** {row['score']:.1%}")
        st.divider()

st.info("Demo dataset is synthetic. Replace it with real project data for production use.")
