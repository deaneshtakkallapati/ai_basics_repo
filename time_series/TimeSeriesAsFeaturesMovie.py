import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

np.random.seed(0)

# Generate synthetic data for 2 years (24 months)
dates = pd.date_range(start='2020-01-01', periods=24, freq='M')
box_office_revenue = 100 + np.sin(np.arange(24) * np.pi / 6) * 50 + np.random.randn(24) * 10

# Create DataFrame
df = pd.DataFrame({'Date': dates, 'Box_Office_Revenue_Millions': box_office_revenue})

# Display the first few rows of the dataframe
print(df.head())

# Extract time series features
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['Year'] = df['Date'].dt.year

# Drop the original date column
df = df.drop(columns=['Date'])

# Selecting features and target variable
X = df[['Month', 'Quarter', 'Year']]
y = df['Box_Office_Revenue_Millions']

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
