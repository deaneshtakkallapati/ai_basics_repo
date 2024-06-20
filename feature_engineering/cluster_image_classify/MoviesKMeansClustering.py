import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
data = {'Budget': [1000000, 2500000, 500000, 3000000, 1500000, 2000000, 1800000, 3500000],
        'Runtime': [120, 110, 95, 130, 105, 115, 100, 125],
        'Rating': [7.5, 8.2, 6.9, 7.8, 7.2, 8.1, 7.0, 8.5],
        'Revenue': [1900000, 3500000, 590000, 3900000, 2500000, 3000000, 2800000, 4500000]
        }
df = pd.DataFrame(data)

scaler = StandardScaler(); scaled_features = scaler.fit_transform(df)
k=3; kmeans = KMeans(n_clusters=3, random_state=42); kmeans.fit(df)
df['Cluster'] = kmeans.fit_predict(df)

plt.figure(figsize=(8, 8))
sns.scatterplot(x='Runtime', y='Revenue', hue='Cluster', data=df, palette='viridis', s=100, alpha=0.8, legend='full')
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 3], marker='X', c='red', s=200, label='Cluster Centers')
plt.title('KMeans Cluster'); plt.xlabel('Budget'); plt.ylabel('Runtime'); plt.legend(); plt.grid(True)
plt.show()
