import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
ewa1 = pd.read_csv("E:/360 for classes/360 digit Assignments/dataset assignment clus/EastWestAirlinescsv.csv")

ewa1.describe()
ewa1.info()

ewa = ewa1.drop(["ID#"], axis=1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(ewa.iloc[:, 0:])
df_norm.describe()

#finding outliers in eastwestairlines#

# Detection of outliers (find limits for salary based on IQR)
IQR = ewa['Balance'].quantile(0.75) - ewa['Balance'].quantile(0.25)
lower_limit = ewa['Balance'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Balance'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa.Balance)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Balance'])

ewa['Balance'] = winsor.fit_transform(ewa[['Balance']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa.Balance)

# Detection of outliers (find limits for salary based on IQR)
IQR = ewa['Bonus_miles'].quantile(0.75) - ewa['Bonus_miles'].quantile(0.25)
lower_limit = ewa['Bonus_miles'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Bonus_miles'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa.Qual_miles)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Bonus_miles'])

ewa['Bonus_miles'] = winsor.fit_transform(ewa[['Bonus_miles']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa.Bonus_miles)


# Detection of outliers (find limits for salary based on IQR)
IQR = ewa['Bonus_trans'].quantile(0.75) - ewa['Bonus_trans'].quantile(0.25)
lower_limit = ewa['Bonus_trans'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Bonus_trans'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa.Qual_miles)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Bonus_trans'])

ewa['Bonus_trans'] = winsor.fit_transform(ewa[['Bonus_trans']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa.Bonus_trans)

# Detection of outliers (find limits for salary based on IQR)
IQR = ewa['Flight_miles_12mo'].quantile(0.75) - ewa['Flight_miles_12mo'].quantile(0.25)
lower_limit = ewa['Flight_miles_12mo'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Flight_miles_12mo'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa.Qual_miles)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Flight_miles_12mo'])

ewa['Flight_miles_12mo'] = winsor.fit_transform(ewa[['Flight_miles_12mo']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa.Flight_miles_12mo)


# Detection of outliers (find limits for salary based on IQR)
IQR = ewa['Flight_trans_12'].quantile(0.75) - ewa['Flight_trans_12'].quantile(0.25)
lower_limit = ewa['Flight_trans_12'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Flight_trans_12'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa.Qual_miles)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Flight_trans_12'])

ewa['Flight_trans_12'] = winsor.fit_transform(ewa[['Flight_trans_12']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa.Flight_trans_12)


# outlier analysis for the dataset 
IQR = ewa['Days_since_enroll'].quantile(0.75) - ewa['Days_since_enroll'].quantile(0.25)
lower_limit = ewa['Days_since_enroll'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Days_since_enroll'].quantile(0.75) + (IQR * 1.5)
sns.boxplot(ewa.Qual_miles)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Days_since_enroll'])

ewa['Days_since_enroll'] = winsor.fit_transform(ewa[['Days_since_enroll']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa.Days_since_enroll)






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

ewa1 = ewa.iloc[:, [11,0,1,2,3,4,5,6,7,8,9,10]]
ewa1.head()

# Aggregate mean of each cluster
ewa1.iloc[:, 0:].groupby(ewa1['clust']).mean()

# creating a csv file 
ewa1.to_csv("EastWestAirlinescsv.csv", encoding = "utf-8")

import os
os.getcwd()
