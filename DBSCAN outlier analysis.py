import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("Athelete Data Filtered.xlsx")
df = df.iloc[:, [6, 7]].values
df = pd.DataFrame(df)


# Visualizing the above imported data
# plt.scatter(df[:, 0], df[:, 1], s=10, c='black')
# plt.show()

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,
    init = 'k-means++', max_iter=300, n_init=10)

    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

# Plotting Elbow method graph to know the optimum number of clusters to be made
plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS score")
# plt.show()


from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=3, min_samples=3)

# saved number of clusters in this labels variable and number_of_clusters variable
labels = dbscan.fit_predict(df)
number_of_clusters = np.unique(labels)
print(number_of_clusters)



