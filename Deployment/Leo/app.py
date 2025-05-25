from flask import Flask, render_template, request
import pickle
import pandas as pd
from utils import hybrid_recommendation,fetch_book_details

app = Flask(__name__)

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
    book_details = book_details.reset_index(drop=True).to_dict(orient="index")

    return render_template("recommendations.html", books=book_details)
@app.route("/test-files")
def test_files():
    from utils import svd_model, tfidf_matrix, indices, merged_df, books_df, cf_trainset

    results = {
        "SVD Model Loaded": isinstance(svd_model, object),
        "TF-IDF Matrix Loaded": hasattr(tfidf_matrix, 'shape'),
        "Indices Loaded": isinstance(indices, dict),
        "Merged DataFrame Loaded": not merged_df.empty,
        "Books DataFrame Loaded": not books_df.empty,
        "Collaborative Trainset Loaded": hasattr(cf_trainset, 'all_users'),
    }

    return "<br>".join([f"{k}: {v}" for k, v in results.items()])


if __name__ == "__main__":
    app.run(debug=True)

