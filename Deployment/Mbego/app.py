from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity
from build_model import build_hybrid_model  # Make sure this is correct based on your structure

app = Flask(__name__)

# Load the model (assuming you've saved it after training)
model = build_hybrid_model(df)  # Assuming df is your dataframe

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['user_id']
    if not user_input.isdigit():
        return "❌ Invalid input. User ID should be numeric."
    
    user_id = int(user_input)
    if user_id not in model['user_rated']:
        return "❌ Invalid user ID."
    
    # Get recommendations for the user
    recs = hybrid_recommend(user_id, model)
    if recs.empty:
        return "😕 You seem new. Try cold-start options."
    
    # Display top recommendations
    recommendations = recs[['Title', 'Author', 'Hybrid_Score']].to_dict(orient='records')
    return render_template('recommendations.html', recommendations=recommendations)

@app.route('/similar', methods=['POST'])
def similar_books():
    title = request.form['book_title']
    similar_books = get_similar_books(title, model)
    return render_template('similar_books.html', books=similar_books)

if __name__ == "__main__":
    app.run(debug=True)
