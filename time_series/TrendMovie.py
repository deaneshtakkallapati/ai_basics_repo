import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
To analyze the trend of a movie variable over time, such as box office revenue, IMDb rating, or budget, 
you can use time series visualization techniques in Python. Here, I'll demonstrate 
how to plot the trend of box office revenue for movies over time using synthetic data.
'''
np.random.seed(0)

num_movies = 100
start_date = pd.to_datetime('2020-01-01')
end_date = pd.to_datetime('2020-12-31')

# Generate synthetic time series data
dates = pd.date_range(start=start_date, end=end_date, freq='D')
box_office_revenue = np.random.randint(1, 100, size=len(dates)).cumsum()

# Create DataFrame
df = pd.DataFrame({'Date': dates, 'Box_Office_Revenue_Millions': box_office_revenue})

# Display the first few rows of the dataframe
print(df.head())

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Box_Office_Revenue_Millions'], marker='o', linestyle='-')
plt.title('Trend of Box Office Revenue for Movies')
plt.xlabel('Date')
plt.ylabel('Box Office Revenue (Millions)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


