# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:18:37 2022

@author: USER
"""

import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
ewa1 = pd.read_csv("E:/Data Science 18012022/Hierarcial Clustering/crime_data.csv")

ewa1.describe()
ewa1.info()

ewa = ewa1.drop(["State"], axis=1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(ewa.iloc[:, 0:])
df_norm.describe()

#finding outliers in eastwestairlines#

# Detection of outliers (find limits for salary based on IQR)
IQR = ewa['Murder'].quantile(0.75) - ewa['Murder'].quantile(0.25)
lower_limit = ewa['Murder'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Murder'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa.Murder)


# Detection of outliers (find limits for salary based on IQR)
IQR = ewa['Assault'].quantile(0.75) - ewa['Assault'].quantile(0.25)
lower_limit = ewa['Assault'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Assault'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa.Assault)



# Detection of outliers (find limits for salary based on IQR)
IQR = ewa['UrbanPop'].quantile(0.75) - ewa['UrbanPop'].quantile(0.25)
lower_limit = ewa['UrbanPop'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['UrbanPop'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa.UrbanPop)



# Detection of outliers (find limits for salary based on IQR)
IQR = ewa['Rape'].quantile(0.75) - ewa['Rape'].quantile(0.25)
lower_limit = ewa['Rape'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Rape'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa.Rape)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Rape'])

ewa['Rape'] = winsor.fit_transform(ewa[['Rape']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa.Rape)





# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 20));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('index');plt.ylabel('distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

ewa['clust'] = cluster_labels # creating a new column and assigning it to new column 

ewa1 = ewa.iloc[:, [4,0,1,2,3]]
ewa1.head()

# Aggregate mean of each cluster
ewa1.iloc[:, 0:].groupby(ewa1['clust']).mean()

# creating a csv file 
ewa1.to_csv("crime_data.csv", encoding = "utf-8")

import os
os.getcwd()
0