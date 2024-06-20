import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import shap

np.random.seed(0)
plt.title('SHAP Values for Movie Attributes')
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

# Explain the model's predictions using SHAP
explainer = shap.Explainer(gb_regressor, X_train)
shap_values = explainer(X_test)

# Summary plot of SHAP values
shap.summary_plot(shap_values, X_test)


# Force plot for a single prediction (change index as needed)
# Initialize JavaScript for SHAP visualization
shap.initjs()

# Specify which feature to explain (change index as needed)
sample_idx = 0

# Force plot for the selected instance
shap.force_plot(explainer.expected_value, shap_values.values[sample_idx], X_test.iloc[sample_idx, :])

