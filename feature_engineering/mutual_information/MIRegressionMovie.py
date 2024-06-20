import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic movie data with features and revenue as the target variable
np.random.seed(0)

# Sample movie data
num_movies = 1000
data = {
    'Marketing_Budget': np.random.randint(100, 1000, size=num_movies),
    'Cast_Expenses': np.random.randint(100, 1000, size=num_movies),
    'Director_Salary': np.random.randint(100, 10000, size=num_movies),
    'Screen_Count': np.random.randint(100, 5000, size=num_movies),
    'Release_Year': np.random.randint(1990, 2023, size=num_movies),
    'Duration_Minutes': np.random.randint(60, 240, size=num_movies),
    'Awards_Count': np.random.randint(0, 5, size=num_movies),
    'Director_Popularity': np.random.randint(1, 10, size=num_movies),
    'Box_Office_Revenue_Millions': np.random.randint(1, 1000, size=num_movies)
}

# Create DataFrame
df = pd.DataFrame(data)

# Encode categorical variables if needed (for simplicity, we'll skip encoding in this example)

# Separate features and target variable
X = df.drop(columns=['Box_Office_Revenue_Millions'])  # Features
y = df['Box_Office_Revenue_Millions']  # Target variable

# Scale features to [0, 1] range (important for mutual info regression)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Calculate mutual information between each feature and the target
mi_scores = mutual_info_regression(X_scaled, y)
print(mi_scores)

# Plotting the mutual information scores
plt.figure(figsize=(10, 6))
plt.barh(range(len(mi_scores)), mi_scores, align='center')
plt.yticks(range(len(mi_scores)), X.columns)
plt.xlabel('Mutual Information Scores')
plt.ylabel('Features')
plt.title('Mutual Information Regression Scores for Movie Features')
plt.tight_layout()
plt.show()
