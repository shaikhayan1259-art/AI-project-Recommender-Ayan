import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ProjectRecommender:
    def __init__(self, csv_path):
        self.projects = pd.read_csv(csv_path).fillna("")
        self.projects["text"] = (
            self.projects["title"] + " " +
            self.projects["description"] + " " +
            self.projects["domain"] + " " +
            self.projects["skills"] + " " +
            self.projects["career_area"]
        )
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.projects["text"])

    def recommend(self, profile_text, top_n=5, domain=None, difficulty=None):
        profile_vector = self.vectorizer.transform([profile_text])
        scores = cosine_similarity(profile_vector, self.matrix).flatten()

        result = self.projects.copy()
        result["score"] = scores

        if domain and domain != "All":
            result = result[result["domain"].str.lower() == domain.lower()]
        if difficulty and difficulty != "All":
            result = result[result["difficulty"].str.lower() == difficulty.lower()]

        return result.sort_values("score", ascending=False).head(top_n)
