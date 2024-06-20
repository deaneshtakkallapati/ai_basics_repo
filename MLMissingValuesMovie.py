import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

'''
Handling missing values is an essential part of data preprocessing in any machine learning project. 
Let's create a sample example using Python to demonstrate how to handle missing values in a 
dataset of movie attributes.
'''

# Generate synthetic movie data with missing values
np.random.seed(0)

num_movies = 1000

# Sample movie data with missing values
data = {
    'Title': [f'Movie {i+1}' for i in range(num_movies)],
    'Genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Thriller'], size=num_movies),
    'Director': np.random.choice(['Director A', 'Director B', 'Director C', 'Director D', 'Director E', np.nan], size=num_movies),
    'Production_Company': np.random.choice(['Company X', 'Company Y', 'Company Z', np.nan], size=num_movies),
    'Budget_Millions': np.random.randint(1, 100, size=num_movies),
    'Box_Office_Revenue_Millions': np.random.randint(1, 1000, size=num_movies),
    'Release_Date': pd.date_range(start='2010-01-01', periods=num_movies, freq='D'),
    'Duration_Minutes': np.random.randint(60, 240, size=num_movies),
    'Rating': np.random.uniform(1, 10, size=num_movies).round(1),
    'Language': np.random.choice(['English', 'French', 'Spanish', 'German', np.nan], size=num_movies),
    'Country': np.random.choice(['USA', 'UK', 'France', 'Germany', 'Canada', np.nan], size=num_movies),
    'Awards_Won': np.random.randint(0, 5, size=num_movies),
    'IMDB_Rating': np.random.uniform(1, 10, size=num_movies).round(1)
}

# Introducing missing values
for col in ['Director', 'Production_Company', 'Language', 'Country']:
    data[col] = np.where(np.random.rand(num_movies) < 0.1, np.nan, data[col])

# Create DataFrame
df = pd.DataFrame(data)

# Drop rows with any missing values
df_dropna = df.dropna()

# Fill missing values in numerical columns with mean
df['Budget_Millions'].fillna(df['Budget_Millions'].mean(), inplace=True)

# Fill missing values in categorical columns with the most frequent value
for col in ['Director', 'Production_Company', 'Language', 'Country']:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Display the first few rows after filling missing values
print("\nDataFrame after filling missing values in categorical columns:")
print(df.head())


