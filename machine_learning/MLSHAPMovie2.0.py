import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import shap

# Generate synthetic data
np.random.seed(0)
num_movies = 1000
data = {
    'Title': [f'Movie {i+1}' for i in range(num_movies)],
    'Genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Thriller'], size=num_movies),
    'Budget_Millions': np.random.randint(1, 100, size=num_movies),
    'Box_Office_Revenue_Millions': np.random.randint(1, 1000, size=num_movies),
    'Duration_Minutes': np.random.randint(60, 240, size=num_movies),
    'Rating': np.random.uniform(1, 10, size=num_movies).round(1),
    'Awards_Won': np.random.randint(0, 5, size=num_movies),
    'IMDB_Rating': np.random.uniform(1, 10, size=num_movies).round(1)
}

df = pd.DataFrame(data)

# Select features and target variable
X = df[['Budget_Millions', 'Duration_Minutes', 'Rating', 'Awards_Won', 'IMDB_Rating']]
y = df['Box_Office_Revenue_Millions']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Gradient Boosting Regressor
gb_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_regressor.fit(X_train, y_train)

# Explain the model's predictions using SHAP
explainer = shap.Explainer(gb_regressor, X_train)
shap_values = explainer(X_test)

# Initialize JavaScript for SHAP visualization
shap.initjs()

# Choose a sample instance to explain (change index as needed)
sample_idx = 0

# Calculate the base value (expected value of model predictions)
base_value = gb_regressor.predict(np.array(X_train.mean()).reshape(1, -1))[0]

# Visualize the SHAP values with force plot
shap.force_plot(base_value, shap_values.values[sample_idx], X_test.iloc[sample_idx, :])
