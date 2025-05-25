from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from fuzzywuzzy import process

app = Flask(__name__)

# Load the pickled model
with open(r'N:\Moringa\afterM\Online book reco\Online-Book-Recommender-System\Deployment\Mbego\recommendation_modeltest.pkl', 'rb') as f:
    model = pickle.load(f)

def hybrid_recommend(user_id, model, top_n=10):
    # Replace with your actual hybrid logic. Here's a placeholder:
    recommendations = model['hybrid_recommendations'].get(user_id, pd.DataFrame())
    return recommendations.head(top_n)

def cold_start_recommend(model, top_n=10):
    """
    Recommend top books based on average rating (or another heuristic).
    Assumes model['book_data'] has 'Title', 'Author', 'AverageRating'.
    """
    book_data = model['book_data']
    
    if 'AverageRating' in book_data.columns:
        top_books = book_data.sort_values(by='AverageRating', ascending=False).head(top_n)
    else:
        # Fallback if no rating column
        top_books = book_data.head(top_n)
    
    return top_books[['Title', 'Author']].to_dict(orient='records')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['user_id']
    if not user_input.isdigit():
        return "❌ Invalid input. User ID should be numeric."
    
    user_id = int(user_input)

    # New or unknown user — show cold-start recommendations
    if user_id not in model['user_rated']:
        cold_recs = cold_start_recommend(model)
        return render_template('recommendations.html',
                               recommendations=cold_recs,
                               message="🆕 You're new here! Based on popularity/content, we recommend:")

    # Valid user — generate hybrid recommendations
    recs = hybrid_recommend(user_id, model)
    
    if recs.empty:
        cold_recs = cold_start_recommend(model)
        return render_template('recommendations.html',
                               recommendations=cold_recs,
                               message="😕 We couldn't find personal matches. Here's what’s trending:")
    
    recommendations = recs[['Title', 'Author', 'Hybrid_Score']].to_dict(orient='records')
    return render_template('recommendations.html', recommendations=recommendations)

@app.route('/similar', methods=['POST'])
def similar_books():
    title = request.form['book_title']
    similar_books = get_similar_books(title, model)
    return render_template('similar_books.html', books=similar_books)

if __name__ == "__main__":
    app.run(debug=True)
