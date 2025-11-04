import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir("D:\\Manoj\\1ExcelR\\Data")

data = pd.read_csv('loan prediction.csv')

#Adding new column 'Total income'

data['Total Income']= data['ApplicantIncome'] +data['CoapplicantIncome']
data1=data[['Total Income','LoanAmount']]
data1.isnull().sum()
med1=int(data1['LoanAmount'].median())
data1['LoanAmount'].fillna(med1,inplace= True)
X = data1.values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
list1 = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, random_state = 42)
    kmeans.fit(X)
    list1.append(kmeans.inertia_)
plt.plot(range(1, 11), list1,marker = "o" )
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within Cluster Distance')

# Fitting K-Means to the dataset
kmeans1 = KMeans(n_clusters = 5,  random_state = 10,n_init=500)
y_kmeans = kmeans1.fit_predict(X)

data["kmeans"]=y_kmeans
data.replace({'kmeans' : { 0 : 'Red', 1 : 'Blue', 2 : 'Green' ,3 : 'orange', 4 : 'purple'}},inplace=True)
data["kmeans"].value_counts()

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'orange', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'purple', label = 'Cluster 5')
plt.scatter(kmeans1.cluster_centers_[:, 0], kmeans1.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income')
plt.ylabel('Loan Amounnt')
plt.legend()

# Using Seaborn
#import seaborn as sns
#sns.scatterplot('Total Income', 'LoanAmount', data=data, hue='kmeans')

