import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import partial_dependence

'''
Partial dependence plots (PDPs) show the relationship between a feature (or multiple features) and the predicted 
outcome of a machine learning model after accounting for the average effect of all other features. 
They are useful for interpreting the impact of individual features on the model's predictions. 
Here's a sample example of how to create partial dependence plots for predicting box office revenue based on 
movie attributes using Python:
'''
# Generate synthetic movie data
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

# Initialize the Gradient Boosting regressor
gb_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Fit the model on the training data
gb_regressor.fit(X_train, y_train)

# Compute partial dependence manually for 'Budget_Millions'
feature_index = 0  # Index of 'Budget_Millions' in X.columns
grid = np.linspace(min(X_train.iloc[:, feature_index]), max(X_train.iloc[:, feature_index]), num=50)
pdp_values, axes = partial_dependence(gb_regressor, X_train, features=[feature_index], grid_resolution=50)

# Plot partial dependence
plt.figure(figsize=(8, 6))
plt.plot(grid, pdp_values[0], marker='o', color='b', linestyle='-')
plt.xlabel('Budget_Millions')
plt.ylabel('Partial Dependence')
plt.title('Partial Dependence Plot for Budget_Millions')
plt.grid(True)
plt.show()
