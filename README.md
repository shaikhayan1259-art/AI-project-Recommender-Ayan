# Project Topic Recommender

A profile-based recommendation system that suggests project topics using student interests, skills and career goals.

## Features
- Student profile input
- Project dataset
- TF-IDF text vectorization
- Cosine similarity
- Top-N project recommendations
- Match score
- Streamlit web UI
- Synthetic demo dataset

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Algorithm
1. Combine each project's title, description, domain, skills and career area.
2. Convert project text into TF-IDF vectors.
3. Convert the student's profile into a TF-IDF vector.
4. Calculate cosine similarity.
5. Sort projects by similarity score.
6. Display the top recommendations.

## Important
The included dataset is synthetic for demonstration. Replace it with authorized real project data before production use.
