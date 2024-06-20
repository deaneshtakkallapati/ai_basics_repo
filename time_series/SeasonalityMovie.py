import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
To illustrate seasonality in movies data, we can generate synthetic data and demonstrate how to detect and visualize 
seasonal patterns using Python. Seasonality refers to periodic fluctuations or patterns that occur at regular intervals 
over time. Let's create an example focusing on the box office revenue of movies and simulate seasonal effects.
'''
np.random.seed(0)

# Generate synthetic data for 2 years (24 months)
dates = pd.date_range(start='2020-01-01', periods=24, freq='M')
box_office_revenue = 100 + np.sin(np.arange(24) * np.pi / 6) * 50 + np.random.randn(24) * 10

# Create DataFrame
df = pd.DataFrame({'Date': dates, 'Box_Office_Revenue_Millions': box_office_revenue})

# Display the first few rows of the dataframe
print(df.head())

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Box_Office_Revenue_Millions'], marker='o', linestyle='-')
plt.title('Seasonal Trend of Box Office Revenue for Movies')
plt.xlabel('Date')
plt.ylabel('Box Office Revenue (Millions)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

