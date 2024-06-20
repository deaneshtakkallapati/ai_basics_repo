import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate synthetic movie data
np.random.seed(0)

num_movies = 10000

# Sample movie data
data = {
    'Title': [f'Movie {i+1}' for i in range(num_movies)],
    'Genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Thriller'], size=num_movies),
    'Director': np.random.choice(['Director A', 'Director B', 'Director C', 'Director D', 'Director E'], size=num_movies),
    'Production_Company': np.random.choice(['Company X', 'Company Y', 'Company Z'], size=num_movies),
    'Budget_Millions': np.random.randint(1, 100, size=num_movies),
    'Box_Office_Revenue_Millions': np.random.randint(1, 1000, size=num_movies),
    'Release_Date': [datetime(2010, 1, 1) + timedelta(days=np.random.randint(0, 3650)) for _ in range(num_movies)],
    'Duration_Minutes': np.random.randint(60, 240, size=num_movies),
    'Rating': np.random.uniform(1, 10, size=num_movies).round(1),
    'Language': np.random.choice(['English', 'French', 'Spanish', 'German'], size=num_movies),
    'Country': np.random.choice(['USA', 'UK', 'France', 'Germany', 'Canada'], size=num_movies),
    'Awards_Won': np.random.randint(0, 5, size=num_movies),
    'IMDB_Rating': np.random.uniform(1, 10, size=num_movies).round(1)
}

# Create DataFrame
df = pd.DataFrame(data)

# Count movies per genre
genre_counts = df['Genre'].value_counts().reset_index()
genre_counts.columns = ['Genre', 'Genre_Count']

# Merge genre counts back into the original DataFrame
df = pd.merge(df, genre_counts, on='Genre', how='left')

df['Release_Year'] = df['Release_Date'].dt.year
df['Release_Month'] = df['Release_Date'].dt.month

# Display updated dataframe with release year and month
print(df.head())
