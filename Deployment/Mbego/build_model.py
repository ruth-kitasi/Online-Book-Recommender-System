# build_model.py

import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def build_hybrid_model(df, sample_size=10000, cf_weight=0.6):
    df_sampled = df.sample(n=min(sample_size, len(df)), random_state=42)

    # === 1. Collaborative Filtering (SVD) ===
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(df_sampled[['User-ID', 'ISBN', 'Book-Rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD()
    svd.fit(trainset)

    # === 2. Content Features ===
    num_cols = ['Year-Of-Publication']
    cat_cols = ['user_age_group', 'user_country']

    num_data = df_sampled[num_cols].fillna(0)
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num_data) if not num_data.empty else np.zeros((len(df_sampled), 0))

    # Categorical handling with safety for Categorical dtype
    cat_data = pd.DataFrame()
    if all(col in df_sampled.columns for col in cat_cols):
        cat_data = df_sampled[cat_cols].copy()
        for col in cat_cols:
            if pd.api.types.is_categorical_dtype(cat_data[col]):
                if 'unknown' not in cat_data[col].cat.categories:
                    cat_data[col] = cat_data[col].cat.add_categories('unknown')
            cat_data[col] = cat_data[col].fillna('unknown')

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    cat_encoded = encoder.fit_transform(cat_data) if not cat_data.empty else np.zeros((len(df_sampled), 0))

    content_features = hstack([num_scaled, cat_encoded]).tocsr()

    # === 3. Book Meta Info ===
    book_meta = df_sampled.set_index('ISBN')[['Book-Title', 'Book-Author', 'Publisher', 'Avg_Rating']]
    has_avg_rating = 'Avg_Rating' in df_sampled.columns

    # === 4. Nearest Neighbors for content similarity ===
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(content_features)

    # === 5. Track user-rated books ===
    user_rated = {}
    for row in df_sampled.itertuples(index=False):
        user_rated.setdefault(row._1, set()).add(row.ISBN)

    # === 6. Return Model Dictionary ===
    return {
        'svd': svd,
        'book_meta': book_meta,
        'user_rated': user_rated,
        'content_features': content_features,
        'knn': knn,
        'has_avg_rating': has_avg_rating,
        'cf_weight': cf_weight
    }
