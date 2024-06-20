import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
# Generate synthetic movie budget data
np.random.seed(0)  # for reproducibility

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

num_movies = 100
factors = ['Genre', 'Production_Size', 'Marketing_Budget', 'Cast_Expenses', 'Director_Salary',
           'Screen_Count', 'Release_Year', 'Duration_Minutes', 'Awards_Count', 'Director_Popularity']
data = {}
for factor in factors:
    data[factor] = np.random.randint(1, 100, size=num_movies)

data['Budget_Millions'] = np.random.randint(50, 200, size=num_movies)

df = pd.DataFrame(data)

# Select numeric columns for PCA
numeric_cols = ['Marketing_Budget', 'Cast_Expenses', 'Director_Salary', 'Screen_Count',
                'Release_Year', 'Duration_Minutes', 'Awards_Count', 'Director_Popularity']
# Extracting the numeric data
x = df[numeric_cols].values
# Standardize the features
x = StandardScaler().fit_transform(x)
# PCA with 2 components
pca = PCA(n_components=5)
principal_components = pca.fit_transform(x)
# Create DataFrame with principal components
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
# Add Budget (target variable) to the DataFrame
df_pca['Budget_Millions'] = df['Budget_Millions']
explained_variance = pca.explained_variance_ratio_
print(explained_variance)
components = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], index=numeric_cols)
print(components)
y = df['Budget_Millions']


# Scatter plot of principal components
plt.figure(figsize=(10, 8))
plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Budget_Millions'], cmap='viridis', s=50)
plt.colorbar(label='Budget (Millions)')
plt.title('PCA of Movies Based on Budget Factors')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
# plt.show()




























