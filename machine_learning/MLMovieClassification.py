import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

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

# Encoding categorical variables
le = LabelEncoder()
df['Genre_Encoded'] = le.fit_transform(df['Genre'])

# Selecting features and target variable
X = df[['Budget_Millions', 'Duration_Minutes', 'Rating', 'Awards_Won', 'IMDB_Rating']]
y = df['Genre_Encoded']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classification model (Logistic Regression)
model = LogisticRegression(max_iter=1000)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.2f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

