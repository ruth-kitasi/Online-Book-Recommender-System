import numpy as np
import pandas as pd
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity

def hybrid_recommend(user_id, model, n=5):
    book_meta = model['book_meta']
    user_rated = model['user_rated'].get(user_id, set())
    candidates = list(set(book_meta.index) - user_rated)

    if not candidates:
        return pd.DataFrame()

    try:
        cf_scores = np.array([model['svd'].predict(user_id, isbn).est for isbn in candidates])
    except:
        cf_scores = np.ones(len(candidates)) * 3.0

    if user_rated:
        rated_indices = [book_meta.index.get_loc(isbn) for isbn in user_rated if isbn in book_meta.index]
        user_profile = model['content_features'][rated_indices].mean(axis=0).A1 if rated_indices else np.ones(model['content_features'].shape[1])
        candidate_indices = [book_meta.index.get_loc(isbn) for isbn in candidates]
        candidate_vectors = model['content_features'][candidate_indices].toarray()
        content_scores = cosine_similarity([user_profile], candidate_vectors)[0]
    else:
        content_scores = book_meta['AvgRating'].fillna(3.0).values if model['has_avg_rating'] else np.ones(len(candidates))

    alpha = model['cf_weight'] if user_id in model['user_rated'] else 0.3
    hybrid_scores = alpha * cf_scores + (1 - alpha) * content_scores

    top_indices = np.argsort(hybrid_scores)[-n:][::-1]
    return pd.DataFrame([{
        'ISBN': candidates[i],
        'Title': book_meta.loc[candidates[i], 'BookTitle'],
        'Author': book_meta.loc[candidates[i], 'BookAuthor'],
        'Hybrid_Score': round(hybrid_scores[i], 3)
    } for i in top_indices])

def get_similar_books(title, model, top_n=5):
    book_meta = model['book_meta']
    titles = book_meta['BookTitle'].dropna().unique()
    match, _ = process.extractOne(title, titles)
    matched_isbn = book_meta[book_meta['BookTitle'] == match].index[0]
    idx = list(book_meta.index).index(matched_isbn)
    book_vector = model['content_features'][idx]
    distances, indices = model['knn'].kneighbors(book_vector, n_neighbors=top_n + 1)
    similar_isbns = book_meta.iloc[indices[0][1:]]
    return similar_isbns[['BookTitle', 'BookAuthor']].reset_index(drop=True)

def recommend_books(model):
    print("\n📚 Welcome to the Book Recommender!")
    print("Choose an option:")
    print("1. Get book recommendations by User-ID")
    print("2. Find similar books by Book Title")

    choice = input("Enter 1 or 2: ").strip()

    if choice == '1':
        user_input = input("Enter your User-ID: ").strip()
        if not user_input.isdigit():
            print("❌ Invalid input. User ID should be numeric.")
            return

        user_id = int(user_input)
        if user_id not in model['user_rated']:
            print("🆕 New user detected! No prior history found.")
            print("Try getting recommendations by author or publisher.")
            ask = input("Would you like to search by author or publisher? (a/p): ").strip().lower()
            col = 'BookAuthor' if ask == 'a' else 'Publisher'
            keyword = input(f"Enter {col}: ").strip().lower()
            matches = model['book_meta'][model['book_meta'][col].str.lower().str.contains(keyword, na=False)]
            print("\n📖 Based on your input, here are some suggestions:")
            print(matches[['BookTitle', 'BookAuthor']].head(5))
            return

        recs = hybrid_recommend(user_id, model)
        if recs.empty:
            print("😕 You seem new. Try cold-start options.")
            ask = input("Would you like to search by author or publisher? (a/p): ").strip().lower()
            col = 'BookAuthor' if ask == 'a' else 'Publisher'
            keyword = input(f"Enter {col}: ").strip().lower()
            matches = model['book_meta'][model['book_meta'][col].str.lower().str.contains(keyword, na=False)]
            print("\n📖 Based on your input, here are some suggestions:")
            print(matches[['BookTitle', 'BookAuthor']].head(5))
        else:
            print("\n✅ Top Book Recommendations:")
            print(recs[['Title', 'Author', 'Hybrid_Score']])

    elif choice == '2':
        title = input("Enter a book title: ").strip()
        try:
            print("\n📚 Books similar to your choice:")
            similar = get_similar_books(title, model)
            print(similar)
        except:
            print("❌ Could not find similar books.")
    else:
        print("❌ Invalid option.")
