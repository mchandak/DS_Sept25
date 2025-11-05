import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

os.chdir("D:\\Manoj\\1ExcelR\\Data")
df = pd.read_csv('Mall_Customers.csv')

df.shape
df.head(10)

# Select features for clustering
features = ['Age', 'AnnualIncome', 'SpendingScore']
X = df[features].copy()

# Standardize the features (CRITICAL for DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

min_pts = 5
eps_values = 0.5

# Fit final model
dbscan = DBSCAN(eps=eps_values, min_samples=min_pts)
labels = dbscan.fit_predict(X_scaled)

#dbscan.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise} ({(n_noise/len(labels))*100:.2f}%)")

df_clust=pd.DataFrame(labels,columns=['cluster'])
df_clust
print(df_clust['cluster'].value_counts())

df_clust1 = pd.concat([df,df_clust],axis=1)

noisedata = df_clust1[df_clust1['cluster']==-1]
clusterdata = df_clust1[df_clust1['cluster']!=-1]

# Silhouette score
mask = labels != -1
if mask.sum() > 1:
    final_silhouette = silhouette_score(X_scaled[mask], labels[mask])
    print(f"\nFinal Silhouette Score: {final_silhouette:.3f}")





