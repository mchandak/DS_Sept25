import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

os.chdir("D:\\Manoj\\1ExcelR\\Data")
df = pd.read_csv('Movie.csv')

# Create matrix
user_item_matrix = df.pivot_table(index='userId', columns='movie', values='rating')
user_item_matrix_filled = user_item_matrix.fillna(0)

# Compares movies based on how users rated them
item_similarity = cosine_similarity(user_item_matrix_filled.T)
item_similarity_df = pd.DataFrame(item_similarity,
                                  index=user_item_matrix_filled.columns,
                                  columns=user_item_matrix_filled.columns)

def find_similar_movies(movie_name, n=3):
    """Find top N most similar movies"""
    similarities = item_similarity_df[movie_name].sort_values(ascending=False)[1:n+1]
    print(similarities)
    
# Show similar movies
find_similar_movies('Toy Story (1995)', n=3)

def show_user_profile(user_id):
    """Show what movies user has already rated"""
    user_ratings = user_item_matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0].sort_values(ascending=False)
    print(rated_movies)

show_user_profile(12)

def recommend_movies(user_id, n_recommendations=5):
        
    # Get user's ratings
    user_ratings = user_item_matrix_filled.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0]
    
    # Check if user has ratings
    if len(rated_movies) == 0:
        return f"Error: User {user_id} has no ratings"
    
    # Get unrated movies
    unrated_movies = user_ratings[user_ratings == 0].index
            
    # Calculate recommendation scores
    recommendations = {}
    for movie in unrated_movies:
        similarity = item_similarity_df.loc[movie, rated_movies.index]
        score = (similarity * rated_movies.values).mean()
        recommendations[movie] = score
    
    # Get top N recommendations
    top_movies = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    result_df = pd.DataFrame(top_movies, columns=['Movie', 'Score']).reset_index(drop=True)
    result_df.index = result_df.index + 1
    return result_df

# Get recommendations

show_user_profile(12)
print("Top 5 Recommendations:")
recs = recommend_movies(12, n_recommendations=5)
recs
