import pickle,pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- Load models, dataframes, and matrices ---

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

with open("data/user_cf_trainset.pkl", "rb") as f:
    user_cf_trainset = pickle.load(f)

#load content_model_data
content_model_data = books_df[['Book-Title', 'Book-Author', 'Publisher']].copy()


# --- Utility Function ---
def clean_title(title):
    """Cleans book titles by removing 'A Novel' and extra whitespace."""
    return title.lower().replace("a novel", "").strip()

# --- Preprocessing for Content-Based Filtering ---
merged_df['Clean-Title'] = merged_df['Book-Title'].fillna('').apply(clean_title)

# --- Popularity Computations ---
valid_popularity_df = merged_df[(merged_df['Book-Rating'] > 0) & (merged_df['Missing_Metadata'] == False)]
book_avg_ratings = valid_popularity_df.groupby('ISBN')['Book-Rating'].mean()
book_rating_counts = valid_popularity_df.groupby('ISBN').size()
valid_books = book_rating_counts[book_rating_counts >= 5].index
book_avg_ratings = book_avg_ratings.loc[valid_books]
pop_overall = book_avg_ratings.sort_values(ascending=False).index.tolist()

pop_by_age_group = (
    valid_popularity_df[valid_popularity_df['ISBN'].isin(valid_books)]
    .groupby(['Age_Group', 'ISBN'])['Book-Rating']
    .mean().sort_values(ascending=False)
    .groupby('Age_Group')
    .apply(lambda x: x.index.get_level_values(1).tolist())
    .to_dict()
)

pop_by_country = (
    valid_popularity_df[valid_popularity_df['ISBN'].isin(valid_books)]
    .groupby(['Country', 'ISBN'])['Book-Rating']
    .mean().sort_values(ascending=False)
    .groupby('Country')
    .apply(lambda x: x.index.get_level_values(1).tolist())
    .to_dict()
)

pop_by_city = (
    valid_popularity_df[valid_popularity_df['ISBN'].isin(valid_books)]
    .groupby(['City', 'ISBN'])['Book-Rating']
    .mean().sort_values(ascending=False)
    .groupby('City')
    .apply(lambda x: x.index.get_level_values(1).tolist())
    .to_dict()
)

# --- 1. User-Based Collaborative Filtering ---
def get_top_n_user_cf(user_id, n=5):
    try:
        inner_uid = user_cf_trainset.to_inner_uid(user_id)
    except ValueError:
        return []
    if inner_uid not in user_cf_trainset.ur or len(user_cf_trainset.ur[inner_uid]) == 0:
        return []

    rated_items = set(j for (j, _) in user_cf_trainset.ur[inner_uid])
    unseen_items = set(user_cf_trainset.all_items()) - rated_items

    predictions = [svd_model.predict(user_id, user_cf_trainset.to_raw_iid(i)) for i in unseen_items]
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)

    result = []
    for pred in top_n:
        book_row = merged_df[(merged_df['ISBN'] == pred.iid) & (merged_df['Missing_Metadata'] == False)]
        if not book_row.empty:
            result.append(book_row.iloc[0]['Book-Title'])
        if len(result) >= n:
            break
    return result

# --- 2. Content-Based Filtering ---
def recommend_books_content(title, n=5):
    cleaned_input_title = clean_title(title)
    matches = content_model_data[content_model_data['Clean-Title'].str.lower() == cleaned_input_title.lower()]
    if matches.empty:
        return []

    idx = matches.index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = sim_scores.argsort()[::-1][1:]

    recommended_books = []
    for i in similar_indices:
        row = content_model_data.iloc[i]
        title_candidate = row['Book-Title']
        meta_row = merged_df[(merged_df['Book-Title'] == title_candidate) & (merged_df['Missing_Metadata'] == False)]
        if not meta_row.empty:
            recommended_books.append(meta_row.iloc[0]['Book-Title'])
        if len(recommended_books) >= n:
            break
    return recommended_books

# --- 3. Popularity-Based Recommendations ---
def recommend_popular_books(user_id=None, n=5):
    user_row = merged_df[merged_df['User-ID'] == user_id]
    isbns = []

    if not user_row.empty:
        city = user_row.iloc[0]['City']
        country = user_row.iloc[0]['Country']
        age_group = user_row.iloc[0]['Age_Group']

        if city in pop_by_city:
            isbns = pop_by_city[city]
        elif country in pop_by_country:
            isbns = pop_by_country[country]
        elif age_group in pop_by_age_group:
            isbns = pop_by_age_group[age_group]

    if not isbns:
        isbns = pop_overall

    recommended_books = []
    for isbn in isbns:
        book_row = merged_df[(merged_df['ISBN'] == isbn) & (merged_df['Missing_Metadata'] == False)]
        if not book_row.empty:
            recommended_books.append(book_row.iloc[0]['Book-Title'])
        if len(recommended_books) >= n:
            break
    return recommended_books

# --- 4. Hybrid Recommendation ---
def hybrid_recommendation(user_id, favorite_book_title=None, n_cf=3, n_content=2, n_popular=2):
    recommendations = []

    cf_recommendations = get_top_n_user_cf(user_id, n=n_cf)
    recommendations.extend(cf_recommendations)
    got_cf = len(cf_recommendations) > 0

    got_content = False
    if favorite_book_title:
        content_recs = recommend_books_content(favorite_book_title, n=n_content)
        if content_recs:
            recommendations.extend(content_recs)
            got_content = True

    if not got_cf or not got_content or len(recommendations) < (n_cf + n_content):
        extra_needed = (n_cf + n_content + n_popular) - len(recommendations)
        popular_recs = recommend_popular_books(user_id, n=extra_needed)
        recommendations.extend(popular_recs)

    seen = set()
    final_recommendations = []
    for book in recommendations:
        if book and book not in seen:
            seen.add(book)
            final_recommendations.append(book)

    return final_recommendations[:n_cf + n_content + n_popular]

# --- Optional: Fetch Metadata (if used in app.py rendering) ---
def fetch_book_details(book_titles):
    return merged_df[merged_df['Book-Title'].isin(book_titles) & (merged_df['Missing_Metadata'] == False)]
