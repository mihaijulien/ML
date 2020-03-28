import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

cust_df = pd.read_csv("dataset/Cust_Segmentation.csv")
print(cust_df.head())

# Pre-processing
'''
As you can see, Address in this dataset is a categorical variable. k-means algorithm isn't directly applicable to categorical 
variables because Euclidean distance function isn't really meaningful for discrete variables. So, lets drop this feature and run clustering.
'''
df = cust_df.drop('Address', axis=1)
print(df.head())

# Normalizing over the standard deviation
'''
Now let's normalize the dataset. But why do we need normalization in the first place? Normalization is a statistical method 
that helps mathematical-based algorithms to interpret features with different magnitudes and distributions equally. 
We use StandardScaler() to normalize our dataset.
'''
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)

# Modelling
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(Clus_dataSet)
labels = k_means.labels_
print(labels)

# Insights
# We assign the labels to each row in dataframe.
df["Clus_km"] = labels
df.head(5)

# Check the centroid values by averaging the features in each cluster.
df.groupby('Clus_km').mean()

area = np.pi * ( X[:, 1])**2
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()

fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))

'''
k-means will partition your customers into mutually exclusive groups, for example, into 3 clusters. The customers in each cluster 
are similar to each other demographically. Now we can create a profile for each group, considering the common characteristics of each cluster. 

For example, the 3 clusters can be:

AFFLUENT, EDUCATED AND OLD AGED
MIDDLE AGED AND MIDDLE INCOME
YOUNG AND LOW INCOME
'''