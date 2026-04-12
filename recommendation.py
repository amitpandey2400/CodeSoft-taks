import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. Dummy Dataset Generation
# ==========================================

# Movie Dataset (For Content-Based Filtering)
movies_data = {
    'movie_id': [1, 2, 3, 4, 5, 6, 7],
    'title': [
        'Inception', 
        'The Matrix', 
        'Interstellar', 
        'The Notebook', 
        'Titanic', 
        'Avengers: Endgame', 
        'Spider-Man: No Way Home'
    ],
    'genres': [
        'Action Sci-Fi Thriller',
        'Action Sci-Fi',
        'Adventure Drama Sci-Fi',
        'Drama Romance',
        'Drama Romance',
        'Action Adventure Sci-Fi',
        'Action Adventure Fantasy'
    ]
}
movies_df = pd.DataFrame(movies_data)

# User Ratings Dataset (For Collaborative Filtering)
# Users: u1, u2, u3, u4
ratings_data = {
    'user_id': [
        'u1', 'u1', 'u1',  # u1 likes Sci-Fi / Action
        'u2', 'u2', 'u2',  # u2 likes Romance / Drama
        'u3', 'u3', 'u3', 'u3', # u3 likes Action / Sci-Fi too (similar to u1)
        'u4', 'u4'         # u4 likes mixed
    ],
    'movie_id': [
        1, 2, 3,
        4, 5, 3,
        1, 2, 6, 7,
        4, 6
    ],
    'rating': [
        5, 4, 5,
        5, 5, 2,
        4, 5, 5, 4,
        4, 3
    ]
}
ratings_df = pd.DataFrame(ratings_data)


# ==========================================
# 2. Content-Based Filtering
# ==========================================
def get_content_based_recommendations(movie_title, top_n=3):
    """
    Recommends movies similar to the given movie_title based on genres.
    """
    if movie_title not in movies_df['title'].values:
        return f"Movie '{movie_title}' not found in the dataset."
    
    # Create the count matrix from the genres
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(movies_df['genres'])
    
    # Compute the Cosine Similarity matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    # Create a Series for movie indices
    indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()
    
    idx = indices[movie_title]
    
    # Get pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top N most similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:top_n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices].tolist()


# ==========================================
# 3. Collaborative Filtering (User-Based)
# ==========================================
def get_collaborative_recommendations(target_user, top_n=2):
    """
    Recommends movies to the target_user based on the preferences of similar users.
    """
    if target_user not in ratings_df['user_id'].values:
        return f"User '{target_user}' not found in the dataset."

    # Create a User-Item matrix
    user_movie_matrix = ratings_df.pivot_table(index='user_id', columns='movie_id', values='rating')
    user_movie_matrix = user_movie_matrix.fillna(0) # Fill NaN with 0 rating
    
    # Compute Cosine Similarity between users
    user_sim = cosine_similarity(user_movie_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_movie_matrix.index, columns=user_movie_matrix.index)
    
    # Get similarity scores for the target user
    sim_scores = user_sim_df[target_user].sort_values(ascending=False)
    
    # Find the most similar user (excluding themselves)
    similar_user = sim_scores.index[1]
    
    # Get movies the similar user has rated highly (> 3)
    similar_user_movies = user_movie_matrix.loc[similar_user]
    top_movies = similar_user_movies[similar_user_movies > 3].index.tolist()
    
    # Get movies the target user has already seen
    target_user_movies = user_movie_matrix.loc[target_user]
    seen_movies = target_user_movies[target_user_movies > 0].index.tolist()
    
    # Recommend movies the similar user liked but the target user hasn't seen
    recommendations = [movie for movie in top_movies if movie not in seen_movies]
    
    # Map movie IDs to Titles
    rec_titles = movies_df[movies_df['movie_id'].isin(recommendations)]['title'].tolist()
    
    return rec_titles[:top_n]


# ==========================================
# 4. Main Execution (Demo)
# ==========================================
if __name__ == "__main__":
    print("-" * 50)
    print("🎬 MOVIE RECOMMENDATION SYSTEM DEMO 🎬")
    print("-" * 50)
    print("\nDataset loaded successfully. Available movies:")
    for title in movies_df['title'].values:
        print(f" - {title}")
        
    print("\n" + "="*50)
    print("1. CONTENT-BASED FILTERING")
    print("="*50)
    target_movie = "The Matrix"
    print(f"Movies similar to '{target_movie}':")
    cb_recs = get_content_based_recommendations(target_movie)
    for i, rec in enumerate(cb_recs, 1):
        print(f"  {i}. {rec}")

    target_movie2 = "The Notebook"
    print(f"\nMovies similar to '{target_movie2}':")
    cb_recs2 = get_content_based_recommendations(target_movie2)
    for i, rec in enumerate(cb_recs2, 1):
        print(f"  {i}. {rec}")

    print("\n" + "="*50)
    print("2. COLLABORATIVE FILTERING (USER-BASED)")
    print("="*50)
    target_user = "u1" 
    print(f"Recommendations for user '{target_user}':")
    print("(User 'u1' likes Action/Sci-Fi. We find similar users and recommend what they liked)")
    collab_recs = get_collaborative_recommendations(target_user)
    if collab_recs:
        for i, rec in enumerate(collab_recs, 1):
            print(f"  {i}. {rec}")
    else:
        print("  No new recommendations found for this user based on similar users.")
        
    # Another example
    target_user2 = "u2"
    print(f"\nRecommendations for user '{target_user2}':")
    print("(User 'u2' likes Romance/Drama)")
    collab_recs2 = get_collaborative_recommendations(target_user2)
    if collab_recs2:
        for i, rec in enumerate(collab_recs2, 1):
            print(f"  {i}. {rec}")
    else:
        print("  No new recommendations found for this user based on similar users.")
    print("\n" + "-" * 50)
