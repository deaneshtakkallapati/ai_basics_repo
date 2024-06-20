import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.feature_selection import mutual_info_regression

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

# Calculate mutual information between each feature and the target
mi_scores = mutual_info_regression(X, y)

# Plotting the mutual information scores
plt.figure(figsize=(10, 6))
plt.bar(range(len(mi_scores)), mi_scores, align='center')
plt.xticks(range(len(mi_scores)), ['Feature ' + str(i+1) for i in range(len(mi_scores))], rotation=45)
plt.xlabel('Features')
plt.ylabel('Mutual Information Scores')
plt.title('Mutual Information Regression Scores for Features')
plt.tight_layout()
plt.show()
