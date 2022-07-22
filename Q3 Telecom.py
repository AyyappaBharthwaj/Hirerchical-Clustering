# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:32:25 2022

@author: USER
"""

import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
ewa1 = pd.read_csv("E:/Data Science 18012022/Hierarcial Clustering/Telco_customer_churn.csv")
ewa1.isnull
ewa1.describe()
ewa1.info()

ewa = ewa1.drop(["Customer_ID","Count","Quarter"], axis=1)


################ Finding Outliers #########################
# Finding Outliers in Avg_Monthly_Long_Distance_Charges
ewa.Avg_Monthly_Long_Distance_Charges

# let's find outliers in Avg_Monthly_Long_Distance_Charges
sns.boxplot(ewa.Avg_Monthly_Long_Distance_Charges)

# No Outliers detected in Avg_Monthly_Long_Distance_Charges

#################################################################

# Finding Outliers in Avg_Monthly_Long_Distance_Charges################

ewa.Avg_Monthly_GB_Download

# let's find outliers in Avg_Monthly_Long_Distance_Charges
sns.boxplot(ewa.Avg_Monthly_GB_Download)

##### Outliers Found in Avg_Monthly_GB_Download#####################

# Detection of outliers (find limits for Avg_Monthly_GB_Download based on IQR)
IQR = ewa['Avg_Monthly_GB_Download'].quantile(0.75) - ewa['Avg_Monthly_GB_Download'].quantile(0.25)
lower_limit = ewa['Avg_Monthly_GB_Download'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Avg_Monthly_GB_Download'].quantile(0.75) + (IQR * 1.5)

# Trimming Technique
# let's flag the outliers in the data set
outliers_ewa = np.where(ewa['Avg_Monthly_GB_Download'] > upper_limit, True, np.where(ewa['Avg_Monthly_GB_Download'] < lower_limit, True, False))
ewa = ewa.loc[~(outliers_ewa), ]
ewa.shape

# let's explore outliers in the trimmed dataset
sns.boxplot(ewa.Avg_Monthly_GB_Download)

### Flagged out Outliers through Trimming Process ##################

# Finding Outliers in Monthly_Charge
ewa.Monthly_Charge

# let's find outliers in Monthly_Charge
sns.boxplot(ewa.Monthly_Charge)
########### No Outliers Found in Monthly_Charge##############

# Finding Outliers in Total_Charges
ewa.Total_Charges

# let's find outliers in Total_Charges
sns.boxplot(ewa.Total_Charges)
########### No Outliers Found in Total_Charges##############

# Finding Outliers in Total_Refunds
ewa.Total_Refunds

# let's find outliers in Total_Refunds
sns.boxplot(ewa.Total_Refunds)
###### Outliers found in Total_Refunds ######################

# Detection of outliers (find limits for Total_Refunds based on IQR)
IQR = ewa['Total_Refunds'].quantile(0.75) - ewa['Total_Refunds'].quantile(0.25)
lower_limit = ewa['Total_Refunds'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Total_Refunds'].quantile(0.75) + (IQR * 1.5)

############### Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Total_Refunds'])

ewa['Total_Refunds'] = winsor.fit_transform(ewa[['Total_Refunds']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa.Total_Refunds)
###############################################################

# Finding Outliers in Total_Extra_Data_Charges
ewa.Total_Extra_Data_Charges

# let's find outliers in Total_Extra_Data_Charges
sns.boxplot(ewa.Total_Extra_Data_Charges)
###### Outliers found in Total_Extra_Data_Charges ######################

# Detection of outliers (find limits for Total_Extra_Data_Charges based on IQR)
IQR = ewa['Total_Extra_Data_Charges'].quantile(0.75) - ewa['Total_Extra_Data_Charges'].quantile(0.25)
lower_limit = ewa['Total_Extra_Data_Charges'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Total_Extra_Data_Charges'].quantile(0.75) + (IQR * 1.5)

############### Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Total_Extra_Data_Charges'])

ewa['Total_Extra_Data_Charges'] = winsor.fit_transform(ewa[['Total_Extra_Data_Charges']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa.Total_Extra_Data_Charges)
#####################################################################
#####################################################################

# Finding Outliers in Total_Long_Distance_Charges
ewa.Total_Long_Distance_Charges

# let's find outliers in Total_Long_Distance_Charges
sns.boxplot(ewa.Total_Long_Distance_Charges)
###### Outliers found in Total_Long_Distance_Charges ######################

IQR = ewa['Total_Long_Distance_Charges'].quantile(0.75) - ewa['Total_Long_Distance_Charges'].quantile(0.25)
lower_limit = ewa['Total_Long_Distance_Charges'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Total_Long_Distance_Charges'].quantile(0.75) + (IQR * 1.5)

############### Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Total_Long_Distance_Charges'])

ewa['Total_Long_Distance_Charges'] = winsor.fit_transform(ewa[['Total_Long_Distance_Charges']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa.Total_Long_Distance_Charges)

###################################################################
###################################################################

# Finding Outliers in Total_Revenue
ewa.Total_Revenue 

# let's find outliers in Total_Long_Distance_Charges
sns.boxplot(ewa.Total_Revenue)
###### Outliers found in Total_Long_Distance_Charges ######################

IQR = ewa['Total_Revenue'].quantile(0.75) - ewa['Total_Revenue'].quantile(0.25)
lower_limit = ewa['Total_Revenue'].quantile(0.25) - (IQR * 1.5)
upper_limit = ewa['Total_Revenue'].quantile(0.75) + (IQR * 1.5)

############### Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Total_Revenue'])

ewa['Total_Revenue'] = winsor.fit_transform(ewa[['Total_Revenue']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(ewa.Total_Revenue)
ewa.shape

############################################################################


ewa.columns # column names
ewa.shape # will give u shape of the dataframe


# Create dummy variables
ewa_new = pd.get_dummies(ewa)
ewa_new_1 = pd.get_dummies(ewa, drop_first = True)
# we have created dummies for all categorical columns

##### One Hot Encoding works
ewa.columns
ewa.shape
df = ewa.iloc[:,0:]
df.shape

from sklearn.preprocessing import OneHotEncoder
# Creating instance of One Hot Encoder
enc = OneHotEncoder() # initializing method

enc_df = pd.DataFrame(enc.fit_transform(df.iloc[:, 0:]).toarray())
enc_df.shape

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(enc_df, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 20));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Customer_Id');plt.ylabel('Churn')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(enc_df) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

ewa['clust'] = cluster_labels # creating a new column and assigning it to new column 

ewa1 = ewa.iloc[:, [27,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]]
ewa1.head()

# Aggregate mean of each cluster
ewa1.iloc[:, 0:].groupby(ewa1['clust']).mean()

# creating a csv file 
ewa1.to_csv("Telecom_Churn.csv", encoding = "utf-8")

import os
os.getcwd()











