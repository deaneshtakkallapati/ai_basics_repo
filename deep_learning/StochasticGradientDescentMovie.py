import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

'''
Stochastic Gradient Descent (SGD) is a popular optimization algorithm used in machine learning for training models, 
particularly when dealing with large datasets. Here, I'll demonstrate how to implement SGD for training a 
linear regression model to predict box office revenue based on movie attributes using Python and scikit-learn.
'''
np.random.seed(0)

num_movies = 1000

# Sample movie data
data = {
    'Title': [f'Movie {i+1}' for i in range(num_movies)],
    'Genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Thriller'], size=num_movies),
    'Director': np.random.choice(['Director A', 'Director B', 'Director C', 'Director D', 'Director E'], size=num_movies),
    'Production_Company': np.random.choice(['Company X', 'Company Y', 'Company Z'], size=num_movies),
    'Budget_Millions': np.random.randint(1, 100, size=num_movies),
    'Box_Office_Revenue_Millions': np.random.randint(1, 1000, size=num_movies),
    'Release_Date': pd.date_range(start='2010-01-01', periods=num_movies, freq='D'),
    'Duration_Minutes': np.random.randint(60, 240, size=num_movies),
    'Rating': np.random.uniform(1, 10, size=num_movies).round(1),
    'Language': np.random.choice(['English', 'French', 'Spanish', 'German'], size=num_movies),
    'Country': np.random.choice(['USA', 'UK', 'France', 'Germany', 'Canada'], size=num_movies),
    'Awards_Won': np.random.randint(0, 5, size=num_movies),
    'IMDB_Rating': np.random.uniform(1, 10, size=num_movies).round(1)
}

# Create DataFrame
df = pd.DataFrame(data)

# Selecting features and target variable
X = df[['Budget_Millions', 'Duration_Minutes', 'Rating', 'Awards_Won', 'IMDB_Rating']]
y = df['Box_Office_Revenue_Millions']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize SGDRegressor with stochastic gradient descent
sgd_regressor = SGDRegressor(max_iter=1000, alpha=0.001, random_state=42)

# Train the model
sgd_regressor.fit(X_train_scaled, y_train)

# Evaluate the model
train_score = sgd_regressor.score(X_train_scaled, y_train)
test_score = sgd_regressor.score(X_test_scaled, y_test)

print(f'Train R^2 score: {train_score:.2f}')
print(f'Test R^2 score: {test_score:.2f}')
