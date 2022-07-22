import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("E:/Data Science 18012022/Hierarcial Clustering/AutoInsurance.csv")
df.describe()
df.dtypes
df.isnull
df.info()

#drop columns 
df.drop(["Customer","State",'Effective To Date'],axis=1, inplace=True)

df.dtypes

#@create dummies
df_new=pd.get_dummies(df)
df_new1=pd.get_dummies(df , drop_first=True)

#Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df_new1.iloc[:, :])
df_norm.describe()

#for creating dendograms
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('auto insurance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing  3 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

df_norm['clust'] = cluster_labels # creating a new column and assigning it to new column 

df.head()
df_norm.head()

df = df_norm.iloc[:, [47,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,]]
df.head()

# Aggregate mean of each cluster
df.iloc[:, 0:].groupby(df_norm.clust).mean()

# creating a csv file 
df.to_csv("Auto Insurance.csv", encoding = "utf-8")

import os
os.getcwd()
