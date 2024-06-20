import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

# Sample movie data with factors influencing budget
data = {
    'Movie': ['A', 'B', 'C', 'D', 'E'],
    'Genre': ['Action', 'Comedy', 'Drama', 'Action', 'Comedy'],
    'Production_Size': ['Large', 'Small', 'Medium', 'Medium', 'Large'],
    'Marketing_Budget': [10, 5, 8, 12, 9],
    'Cast_Expenses': [20, 15, 18, 22, 17],
    'Director_Salary': [5, 3, 4, 6, 4],
    'Budget (Millions)': [100, 80, 60, 110, 95]
}

# Create a DataFrame
df = pd.DataFrame(data)
# Selecting numeric columns for PCA
numeric_cols = ['Director_Salary', 'Marketing_Budget', 'Cast_Expenses']
# Extracting the numeric data
x = df[numeric_cols].values
# Standardize the data
x = StandardScaler().fit_transform(x)

# PCA with the number of components equal to the number of numeric columns
pca = PCA(n_components=len(numeric_cols))
principal_components = pca.fit_transform(x)

# Create DataFrame to visualize principal components and their correlations with original variables
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])

# Add Budget (target variable) to the DataFrame
df_pca['Budget (Millions)'] = df['Budget (Millions)']

# Variance explained by each principal component
explained_variance = pca.explained_variance_ratio_
# Print principal component loadings (correlations between original variables and principal components)
components = pd.DataFrame(pca.components_, columns=numeric_cols, index=['PC1', 'PC2', 'PC3'])
y = df['Budget (Millions)']
mi_scores = make_mi_scores(df[numeric_cols], y, discrete_features=False)
print(mi_scores)





























