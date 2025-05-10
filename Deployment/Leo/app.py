from flask import Flask, render_template, request
import pickle
import pandas as pd
from utils import hybrid_recommendation,fetch_book_details

app = Flask(__name__)

# ---- Load Models & Data ----
with open("models/svd_model.pkl", "rb") as f:
    svd_model = pickle.load(f)

with open("models/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

with open("models/indices.pkl", "rb") as f:
    indices = pickle.load(f)

with open("data/merged_df.pkl", "rb") as f:
    merged_df = pickle.load(f)

with open("data/books_df.pkl", "rb") as f:
    books_df = pickle.load(f)


# ---- Routes ----
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_id = request.form.get("user_id")
        favorite = request.form.get("favorite_book")
        return render_template("recommendations.html", user_id=user_id, favorite=favorite)
    return render_template("index.html")


from utils import hybrid_recommendation, fetch_book_details

@app.route("/recommend", methods=["POST"])
def recommend():
    user_id = request.form.get("user_id")
    title = request.form.get("favorite_book")
    recs = hybrid_recommendation(user_id, title)
    book_details = fetch_book_details(recs)
    return render_template("recommendations.html", books=book_details)

if __name__ == "__main__":
    app.run(debug=True)
