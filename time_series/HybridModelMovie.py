import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

'''
Creating a hybrid forecasting model for predicting movie box office revenue involves integrating multiple forecasting 
techniques to leverage their strengths and improve accuracy. In this example, 
we'll combine a traditional time series approach (ARIMA) with a machine learning model (Linear Regression) using Python.
We'll use synthetic data for demonstration purposes.
'''

np.random.seed(0)

# Generate synthetic data for 3 years (36 months)
dates = pd.date_range(start='2020-01-01', periods=36, freq='M')
box_office_revenue = 100 + np.sin(np.arange(36) * np.pi / 6) * 50 + np.random.randn(36) * 10

# Create DataFrame
df = pd.DataFrame({'Date': dates, 'Box_Office_Revenue_Millions': box_office_revenue})

# Display the first few rows of the dataframe
print(df.head())

# Extract time series features
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['Year'] = df['Date'].dt.year

# Create lagged features (use previous month revenue as feature)
df['Lagged_Revenue'] = df['Box_Office_Revenue_Millions'].shift(1)

# Drop missing values due to lagging
df = df.dropna()

# Display the updated dataframe
print(df.head())

# Selecting features and target variable
X = df[['Month', 'Quarter', 'Year', 'Lagged_Revenue']]
y = df['Box_Office_Revenue_Millions']

# Split data into training and testing sets
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Fit ARIMA model
arima_model = ARIMA(df['Box_Office_Revenue_Millions'], order=(1, 1, 1))  # Example order, tune as needed
arima_fit = arima_model.fit()

# Forecast using ARIMA model
arima_forecast = arima_fit.forecast(steps=len(X_test))  # Forecast as a pandas Series

# Access forecasted values correctly
arima_forecast_values = arima_forecast.values  # or arima_forecast_series.to_numpy()

# Example of printing and accessing values
print(arima_forecast)  # Print to understand the structure
print(arima_forecast_values)  # Print the forecasted values

# Initialize Linear Regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train, y_train)

# Predict on test data
linear_forecast = linear_model.predict(X_test)

# Hybrid forecast (simple average for demonstration)
hybrid_forecast = (arima_forecast + linear_forecast) / 2

# Calculate MSE for each model
mse_arima = mean_squared_error(y_test, arima_forecast)
mse_linear = mean_squared_error(y_test, linear_forecast)
mse_hybrid = mean_squared_error(y_test, hybrid_forecast)

print(f'MSE - ARIMA: {mse_arima:.2f}')
print(f'MSE - Linear Regression: {mse_linear:.2f}')
print(f'MSE - Hybrid Model: {mse_hybrid:.2f}')

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Box_Office_Revenue_Millions'], label='Actual')
plt.plot(df['Date'][train_size:], hybrid_forecast, label='Hybrid Forecast', linestyle='--')
plt.title('Hybrid Forecasting Model for Movie Box Office Revenue')
plt.xlabel('Date')
plt.ylabel('Box Office Revenue (Millions)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
