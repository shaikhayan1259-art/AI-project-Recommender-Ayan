from recommendation_engine import ProjectRecommender

engine = ProjectRecommender("projects.csv")
result = engine.recommend(
    "python machine learning artificial intelligence data science",
    top_n=3
)

assert len(result) == 3
assert "score" in result.columns
print(result[["title", "score"]].to_string(index=False))
print("TEST PASSED")
