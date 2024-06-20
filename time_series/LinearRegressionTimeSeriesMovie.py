import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

'''
Performing linear regression with time series data involves predicting a continuous variable (like box office revenue) 
based on time-dependent features (such as release date). Here, I'll demonstrate how to build a 
simple linear regression model using time series features for predicting movie box office revenue in Python.
'''
np.random.seed(0)

num_movies = 1000

# Sample movie data
data = {
    'Title': [f'Movie {i+1}' for i in range(num_movies)],
    'Release_Date': pd.date_range(start='2010-01-01', periods=num_movies, freq='D'),
    'Budget_Millions': np.random.randint(1, 100, size=num_movies),
    'Box_Office_Revenue_Millions': np.random.randint(1, 1000, size=num_movies),
    'Duration_Minutes': np.random.randint(60, 240, size=num_movies),
    'Rating': np.random.uniform(1, 10, size=num_movies).round(1),
    'Awards_Won': np.random.randint(0, 5, size=num_movies),
    'IMDB_Rating': np.random.uniform(1, 10, size=num_movies).round(1)
}

# Create DataFrame
df = pd.DataFrame(data)

# Selecting features and target variable
X = df[['Release_Date', 'Budget_Millions', 'Duration_Minutes', 'Rating', 'Awards_Won', 'IMDB_Rating']]
y = df['Box_Office_Revenue_Millions']

# Convert Release_Date to ordinal values (numeric representation of dates)
X['Release_Date_Ordinal'] = X['Release_Date'].apply(lambda x: x.toordinal())
X = X.drop(columns=['Release_Date'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate R^2 score (coefficient of determination)
r2_score = model.score(X_test, y_test)
print(f'R^2 Score: {r2_score:.2f}')

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
plt.xlabel('Actual Box Office Revenue (Millions)')
plt.ylabel('Predicted Box Office Revenue (Millions)')
plt.title('Actual vs Predicted Box Office Revenue')
plt.grid(True)
plt.show()
