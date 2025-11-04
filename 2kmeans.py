import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir("D:\\Manoj\\1ExcelR\\Data")
dataset = pd.read_csv("Mall_Customers.csv")    

X = dataset.iloc[:, [3, 4]].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
list1 = []
for i in range(1, 11):
    kmeans1 = KMeans(n_clusters = i, random_state = 0)
    kmeans1.fit(X)
    list1.append(kmeans1.inertia_)
plt.plot(range(1, 11), list1,marker = "o" )
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

# Fitting K-Means to the dataset
kmeans1 = KMeans(n_clusters = 5,  random_state = 0,n_init=500)
y_kmeans = kmeans1.fit_predict(X)
#####

dataset["kmeans"]=y_kmeans
dataset.replace({'kmeans' : { 0 : 'Red', 1 : 'Blue', 2 : 'Green' ,
                             3 : 'Cyan', 4 : 'Magenta'}},inplace=True)
dataset["kmeans"].value_counts()

c1=['Red','Blue','Green','Cyan','Magenta']
# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, 
            c = c1[0], label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100,
            c = c1[1], label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, 
            c = c1[2], label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, 
            c = c1[3], label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, 
            c = c1[4], label = 'Cluster 5')
plt.scatter(kmeans1.cluster_centers_[:, 0], kmeans1.cluster_centers_[:, 1],
                    s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()

# Calculate silhouette score for 
from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(X, y_kmeans)
print(f'\nSilhouette Score: {silhouette_avg:.4f}')

# Interpretation
if silhouette_avg > 0.7:
    print('Clustering quality: STRONG')
elif silhouette_avg > 0.5:
    print('Clustering quality: REASONABLE')
elif silhouette_avg > 0.25:
    print('Clustering quality: WEAK')
else:
    print('Clustering quality: POOR')
