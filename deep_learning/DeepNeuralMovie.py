import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

'''
Building a deep neural network (DNN) for predicting movie-related data involves constructing a model with multiple 
hidden layers using libraries like TensorFlow/Keras. A deep neural network can potentially capture intricate 
patterns and relationships in the data, improving predictive accuracy. 
Here’s a step-by-step example of how to create a deep neural network for predicting box office revenue based on 
movie attributes
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

# Initialize the deep neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(X_train_scaled.shape[1],), activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Display model summary
model.summary()

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on test data
loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f'Test Mean Absolute Error: {mae:.2f}')

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
