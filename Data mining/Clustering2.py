#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Implementation of KMeans Cluster

# In[2]:


class Kmeans:
    '''Implementing Kmeans algorithm.'''

    def __init__(self, n_clusters, max_iter=100, random_state=123):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def initializ_centroids(self, X):
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))
    
    def fit(self, X):
        self.centroids = self.initializ_centroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.compute_sse(X, self.labels, self.centroids)
    
    def predict(self, X):
        distance = self.compute_distance(X, self.centroids)
        return self.find_closest_cluster(distance)


# Implementation of Kmeadin Cluster

# In[3]:


class Kmedian:
    '''Implementing Kmedian algorithm.'''

    def __init__(self, n_clusters, max_iter=100, random_state=123):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def initializ_centroids(self, X):
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            # Change convert The mean into median
            centroids[k, :] = np.median(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))
    
    def fit(self, X):
        self.centroids = self.initializ_centroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.compute_sse(X, self.labels, self.centroids)
    
    def predict(self, X):
        distance = self.compute_distance(X, self.centroids)
        return self.find_closest_cluster(distance)


# Read the Data File

# In[4]:


df_1 = pd.read_csv('veggies.csv')
df_2 = pd.read_csv('animals.csv')
df_3 = pd.read_csv('fruits.csv')
df_4 = pd.read_csv('countries.csv')


# Run The Kmeans Cluster into Datasets (Unnormalize)

# In[5]:


# Dataset 1

k = 9
km_data_1 = Kmeans(n_clusters=k, max_iter=100)
df_1 = StandardScaler().fit_transform(df_1)
km_data_1.fit(df_1)
centroids_data_1 = km_data_1.centroids
predict_data_1 = km_data_1.predict(df_1)


# Dataset 2

km_data_2 = Kmeans(n_clusters=k, max_iter=100)
df_2 = StandardScaler().fit_transform(df_2)
km_data_2.fit(df_2)
centroids_data_2 = km_data_2.centroids
predict_data_2   = km_data_2.predict(df_2)

# Dataset 3

km_data_3 = Kmeans(n_clusters=k, max_iter=100)
df_3 = StandardScaler().fit_transform(df_3)
km_data_3.fit(df_3)
centroids_data_3 = km_data_3.centroids
predict_data_3   = km_data_3.predict(df_3)

# Dataset 4

km_data_4 = Kmeans(n_clusters=k, max_iter=100)
df_4 = StandardScaler().fit_transform(df_4)
km_data_4.fit(df_4)
centroids_data_4 = km_data_4.centroids
predict_data_4   = km_data_4.predict(df_4)


# (Unormalize) Kmean calculate (precision,recall,f1-score)

# In[6]:


# Dataset 1

precision_data_1 = []
recall_data_1    = []
f1_score_data_1  = []
for i in range(0,9):
    cluster_data = df_1[predict_data_1 == i , :]
    total_sum = 0
    max_sum   = 0
    for n in range(len(cluster_data)):
        total_sum = total_sum + np.sum(cluster_data[n])
        max_sum   = max_sum   + np.max(cluster_data[n])
    precision_result = int(abs((max_sum/total_sum) * 100))
    recall_result    = int(abs((max_sum/(total_sum+9)) * 100))
    if ((precision_result+recall_result) == 0):
        f1_score_data_1.append(0)
    else:
        f1_result        = int(2 * ( (precision_result * recall_result) / (precision_result + recall_result )))
        f1_score_data_1.append(f1_result)
    recall_data_1.append(recall_result)
    precision_data_1.append(precision_result)


# Dataset 2

precision_data_2 = []
recall_data_2    = []
f1_score_data_2  = []
for i in range(0,9):
    cluster_data = df_2[predict_data_2 == i , :]
    total_sum = 0
    max_sum   = 0
    for n in range(len(cluster_data)):
        total_sum = total_sum + np.sum(cluster_data[n])
        max_sum   = max_sum   + np.max(cluster_data[n])
    precision_result = int(abs((max_sum/total_sum) *100 ))
    recall_result    = int(abs((max_sum/(total_sum+9)) * 100))
    if ((precision_result+recall_result) == 0):
        f1_score_data_2.append(0)
    else:
        f1_result        = int(2 * ( (precision_result * recall_result) / (precision_result + recall_result )))
        f1_score_data_2.append(f1_result)
    recall_data_2.append(recall_result)
    precision_data_2.append(precision_result)
    

# Dataset 3

precision_data_3 = []
recall_data_3    = []
f1_score_data_3  = []
for i in range(0,9):
    cluster_data = df_2[predict_data_3 == i , :]
    total_sum = 0
    max_sum   = 0
    for n in range(len(cluster_data)):
        total_sum = total_sum + np.sum(cluster_data[n])
        max_sum   = max_sum   + np.max(cluster_data[n])
    precision_result = int(abs((max_sum/total_sum) *100 ))
    recall_result    = int(abs((max_sum/(total_sum+9)) * 100))
    if ((precision_result+recall_result) == 0):
        f1_score_data_3.append(0)
    else:
        f1_result        = int(2 * ( (precision_result * recall_result) / (precision_result + recall_result )))
        f1_score_data_3.append(f1_result)
    recall_data_3.append(recall_result)
    precision_data_3.append(precision_result)

    
# Dataset 4

precision_data_4 = []
recall_data_4    = []
f1_score_data_4  = []
for i in range(0,9):
    cluster_data = df_2[predict_data_4 == i , :]
    total_sum = 0
    max_sum   = 0
    for n in range(len(cluster_data)):
        total_sum = total_sum + np.sum(cluster_data[n])
        max_sum   = max_sum   + np.max(cluster_data[n])
    precision_result = int(abs((max_sum/total_sum) *100 ))
    recall_result    = int(abs((max_sum/(total_sum+9)) * 100))
    if ((precision_result+recall_result) == 0):
        f1_score_data_4.append(0)
    else:
        f1_result        = int(2 * ( (precision_result * recall_result) / (precision_result + recall_result )))
        f1_score_data_4.append(f1_result)
    recall_data_4.append(recall_result)
    precision_data_4.append(precision_result)


# Visulization  (Unnormalize)

# In[7]:


K = [0,1,2,3,4,5,6,7,8]

# Dataset 1

plt.scatter(K[:] , precision_data_1[:] , c='green', label='precision_veggies')
plt.scatter(K[:] , recall_data_1 [:] ,   c='blue', label='recall_veggies')
plt.scatter(K[:] , f1_score_data_1 [:] , c='red', label='f1_score_veggies')
plt.xlim([-1, 9])
plt.ylim([0,100])
plt.legend()
plt.xlabel('Total Number of Cluster')
plt.ylabel('Precision Recall F1_Score')
plt.title('Unnormalize Data Cluster Kmean')
plt.show()


# Dataset 2

plt.scatter(K[:] , precision_data_2[:] , c='green', label='precision_animals')
plt.scatter(K[:] , recall_data_2 [:] ,   c='blue', label='recall_animals')
plt.scatter(K[:] , f1_score_data_2 [:] , c='red', label='f1_score_animals')
plt.xlim([-1, 9])
plt.ylim([0,100])
plt.legend()
plt.xlabel('Total Number of Cluster')
plt.ylabel('Precision Recall F1_Score')
plt.title('Unnormalize Data Cluster Kmean')
plt.show()


# Dataset 3

plt.scatter(K[:] , precision_data_3[:] , c='green', label='precision_fruits')
plt.scatter(K[:] , recall_data_3[:] ,   c='blue', label='recall_fruits')
plt.scatter(K[:] , f1_score_data_3[:] , c='red', label='f1_score_fruits')
plt.xlim([-1, 9])
plt.ylim([0,100])
plt.legend()
plt.xlabel('Total Number of Cluster')
plt.ylabel('Precision Recall F1_Score')
plt.title('Unnormalize Data Cluster Kmean')
plt.show()


# Dataset 4

plt.scatter(K[:] , precision_data_4[:] , c='green', label='precision_countries')
plt.scatter(K[:] , recall_data_4[:] ,   c='blue', label='recall_countries')
plt.scatter(K[:] , f1_score_data_4[:] , c='red', label='f1_score_countries')
plt.xlim([-1, 9])
plt.ylim([0,100])
plt.legend()
plt.xlabel('Total Number of Cluster')
plt.ylabel('Precision Recall F1_Score')
plt.title('Unnormalize Data Cluster Kmean')
plt.show()


# Normalize the Data before apply the Cluster (Kmeans)

# In[8]:


# Dataset 1

minmaxscaler = MinMaxScaler()
normalize_data_1 = minmaxscaler.fit_transform(df_1)

km_data_1 = Kmeans(n_clusters=k, max_iter=100)
km_data_1.fit(normalize_data_1)
centroids_data_1 = km_data_1.centroids
predict_data_1 = km_data_1.predict(normalize_data_1)

# Dataset 2

normalize_data_2 = minmaxscaler.fit_transform(df_2)

km_data_2 = Kmeans(n_clusters=k, max_iter=100)
km_data_2.fit(normalize_data_2)
centroids_data_2 = km_data_2.centroids
predict_data_2   = km_data_2.predict(normalize_data_2)



# Dataset 3

normalize_data_3 = minmaxscaler.fit_transform(df_3)
km_data_3 = Kmeans(n_clusters=k, max_iter=100)
km_data_3.fit(normalize_data_3)
centroids_data_3 = km_data_3.centroids
predict_data_3   = km_data_3.predict(normalize_data_3)

# Dataset 4

normalize_data_4 = minmaxscaler.fit_transform(df_4)
km_data_4 = Kmeans(n_clusters=k, max_iter=100)
km_data_4.fit(normalize_data_4)
centroids_data_4 = km_data_4.centroids
predict_data_4   = km_data_4.predict(normalize_data_4)


# Normalize Calculate (precision , recall , f1_score) kmeans

# In[9]:


# Dataset 1

precision_data_1 = []
recall_data_1    = []
f1_score_data_1  = []
for i in range(0,9):
    cluster_data = df_1[predict_data_1 == i , :]
    total_sum = 0
    max_sum   = 0
    for n in range(len(cluster_data)):
        total_sum = total_sum + np.sum(cluster_data[n])
        max_sum   = max_sum   + np.max(cluster_data[n])
    precision_result = int(abs((max_sum/total_sum) * 100))
    recall_result    = int(abs((max_sum/(total_sum+9)) * 100))
    if ((precision_result+recall_result) == 0):
        f1_score_data_1.append(0)
    else:
        f1_result        = int(2 * ( (precision_result * recall_result) / (precision_result + recall_result )))
        f1_score_data_1.append(f1_result)
    recall_data_1.append(recall_result)
    precision_data_1.append(precision_result)
print(precision_data_1)

# Dataset 2

precision_data_2 = []
recall_data_2    = []
f1_score_data_2  = []
for i in range(0,9):
    cluster_data = df_2[predict_data_2 == i , :]
    total_sum = 0
    max_sum   = 0
    for n in range(len(cluster_data)):
        total_sum = total_sum + np.sum(cluster_data[n])
        max_sum   = max_sum   + np.max(cluster_data[n])
    precision_result = int(abs((max_sum/total_sum) *100 ))
    recall_result    = int(abs((max_sum/(total_sum+9)) * 100))
    if ((precision_result+recall_result) == 0):
        f1_score_data_2.append(0)
    else:
        f1_result        = int(2 * ( (precision_result * recall_result) / (precision_result + recall_result )))
        f1_score_data_2.append(f1_result)
    recall_data_2.append(recall_result)
    precision_data_2.append(precision_result)
    

# Dataset 3

precision_data_3 = []
recall_data_3    = []
f1_score_data_3  = []
for i in range(0,9):
    cluster_data = df_2[predict_data_3 == i , :]
    total_sum = 0
    max_sum   = 0
    for n in range(len(cluster_data)):
        total_sum = total_sum + np.sum(cluster_data[n])
        max_sum   = max_sum   + np.max(cluster_data[n])
    precision_result = int(abs((max_sum/total_sum) *100 ))
    recall_result    = int(abs((max_sum/(total_sum+9)) * 100))
    if ((precision_result+recall_result) == 0):
        f1_score_data_3.append(0)
    else:
        f1_result        = int(2 * ( (precision_result * recall_result) / (precision_result + recall_result )))
        f1_score_data_3.append(f1_result)
    recall_data_3.append(recall_result)
    precision_data_3.append(precision_result)

    
# Dataset 4

precision_data_4 = []
recall_data_4    = []
f1_score_data_4  = []
for i in range(0,9):
    cluster_data = df_2[predict_data_4 == i , :]
    total_sum = 0
    max_sum   = 0
    for n in range(len(cluster_data)):
        total_sum = total_sum + np.sum(cluster_data[n])
        max_sum   = max_sum   + np.max(cluster_data[n])
    precision_result = int(abs((max_sum/total_sum) *100 ))
    recall_result    = int(abs((max_sum/(total_sum+9)) * 100))
    if ((precision_result+recall_result) == 0):
        f1_score_data_4.append(0)
    else:
        f1_result        = int(2 * ( (precision_result * recall_result) / (precision_result + recall_result )))
        f1_score_data_4.append(f1_result)
    recall_data_4.append(recall_result)
    precision_data_4.append(precision_result)


# Visulization (Normalize)

# In[10]:


# Dataset 1

plt.scatter(K[:] , precision_data_1[:] , c='green', label='precision_veggies')
plt.scatter(K[:] , recall_data_1 [:] ,   c='blue', label='recall_veggies')
plt.scatter(K[:] , f1_score_data_1 [:] , c='red', label='f1_score_veggies')
plt.xlim([-1, 9])
plt.ylim([0,100])
plt.legend()
plt.xlabel('Total Number of Cluster')
plt.ylabel('Precision Recall F1_Score')
plt.title('Normalize Data Cluster Kmean')
plt.show()


# Dataset 2

plt.scatter(K[:] , precision_data_2[:] , c='green', label='precision_animals')
plt.scatter(K[:] , recall_data_2 [:] ,   c='blue', label='recall_animals')
plt.scatter(K[:] , f1_score_data_2 [:] , c='red', label='f1_score_animals')
plt.xlim([-1, 9])
plt.ylim([0,100])
plt.legend()
plt.xlabel('Total Number of Cluster')
plt.ylabel('Precision Recall F1_Score')
plt.title('Normalize Data Cluster Kmean')
plt.show()


# Dataset 3

plt.scatter(K[:] , precision_data_3[:] , c='green', label='precision_fruits')
plt.scatter(K[:] , recall_data_3[:] ,   c='blue', label='recall_fruits')
plt.scatter(K[:] , f1_score_data_3[:] , c='red', label='f1_score_fruits')
plt.xlim([-1, 9])
plt.ylim([0,100])
plt.legend()
plt.xlabel('Total Number of Cluster')
plt.ylabel('Precision Recall F1_Score')
plt.title('Normalize Data Cluster Kmean')
plt.show()


# Dataset 4

plt.scatter(K[:] , precision_data_4[:] , c='green', label='precision_countries')
plt.scatter(K[:] , recall_data_4[:] ,   c='blue', label='recall_countries')
plt.scatter(K[:] , f1_score_data_4[:] , c='red', label='f1_score_countries')
plt.xlim([-1, 9])
plt.ylim([0,100])
plt.legend()
plt.xlabel('Total Number of Cluster')
plt.ylabel('Precision Recall F1_Score')
plt.title('Normalize Data Cluster Kmean')
plt.show()


# Run the Kmedian cluster into Datasets  (Unnormalize)

# In[11]:


# Dataset 1

k = 9
km_data_1 = Kmedian(n_clusters=k, max_iter=100)
df_1 = StandardScaler().fit_transform(df_1)
km_data_1.fit(df_1)
centroids_data_1 = km_data_1.centroids

# Dataset 2

km_data_2 = Kmedian(n_clusters=k, max_iter=100)
df_2 = StandardScaler().fit_transform(df_2)
km_data_2.fit(df_2)
centroids_data_2 = km_data_2.centroids

# Dataset 3

km_data_3 = Kmedian(n_clusters=k, max_iter=100)
df_3 = StandardScaler().fit_transform(df_3)
km_data_3.fit(df_3)
centroids_data_3 = km_data_3.centroids

# Dataset 4

km_data_4 = Kmedian(n_clusters=k, max_iter=100)
df_4 = StandardScaler().fit_transform(df_4)
km_data_4.fit(df_4)
centroids_data_4 = km_data_4.centroids


# Kmedian calculate (precision,recall,f1-score)

# In[12]:


# Dataset 1

precision_data_1 = []
recall_data_1    = []
f1_score_data_1  = []
for i in range(0,9):
    cluster_data = df_1[predict_data_1 == i , :]
    total_sum = 0
    max_sum   = 0
    for n in range(len(cluster_data)):
        total_sum = total_sum + np.sum(cluster_data[n])
        max_sum   = max_sum   + np.max(cluster_data[n])
    precision_result = int(abs((max_sum/total_sum) * 100))
    recall_result    = int(abs((max_sum/(total_sum+9)) * 100))
    if ((precision_result+recall_result) == 0):
        f1_score_data_1.append(0)
    else:
        f1_result        = int(2 * ( (precision_result * recall_result) / (precision_result + recall_result )))
        f1_score_data_1.append(f1_result)
    recall_data_1.append(recall_result)
    precision_data_1.append(precision_result) 

# Dataset 2

precision_data_2 = []
recall_data_2    = []
f1_score_data_2  = []
for i in range(0,9):
    cluster_data = df_2[predict_data_2 == i , :]
    total_sum = 0
    max_sum   = 0
    for n in range(len(cluster_data)):
        total_sum = total_sum + np.sum(cluster_data[n])
        max_sum   = max_sum   + np.max(cluster_data[n])
    precision_result = int(abs((max_sum/total_sum) *100 ))
    recall_result    = int(abs((max_sum/(total_sum+9)) * 100))
    if ((precision_result+recall_result) == 0):
        f1_score_data_2.append(0)
    else:
        f1_result        = int(2 * ( (precision_result * recall_result) / (precision_result + recall_result )))
        f1_score_data_2.append(f1_result)
    recall_data_2.append(recall_result)
    precision_data_2.append(precision_result)
    

# Dataset 3

precision_data_3 = []
recall_data_3    = []
f1_score_data_3  = []
for i in range(0,9):
    cluster_data = df_2[predict_data_3 == i , :]
    total_sum = 0
    max_sum   = 0
    for n in range(len(cluster_data)):
        total_sum = total_sum + np.sum(cluster_data[n])
        max_sum   = max_sum   + np.max(cluster_data[n])
    precision_result = int(abs((max_sum/total_sum) *100 ))
    recall_result    = int(abs((max_sum/(total_sum+9)) * 100))
    if ((precision_result+recall_result) == 0):
        f1_score_data_3.append(0)
    else:
        f1_result        = int(2 * ( (precision_result * recall_result) / (precision_result + recall_result )))
        f1_score_data_3.append(f1_result)
    recall_data_3.append(recall_result)
    precision_data_3.append(precision_result)

    
# Dataset 4

precision_data_4 = []
recall_data_4    = []
f1_score_data_4  = []
for i in range(0,9):
    cluster_data = df_2[predict_data_4 == i , :]
    total_sum = 0
    max_sum   = 0
    for n in range(len(cluster_data)):
        total_sum = total_sum + np.sum(cluster_data[n])
        max_sum   = max_sum   + np.max(cluster_data[n])
    precision_result = int(abs((max_sum/total_sum) *100 ))
    recall_result    = int(abs((max_sum/(total_sum+9)) * 100))
    if ((precision_result+recall_result) == 0):
        f1_score_data_4.append(0)
    else:
        f1_result        = int(2 * ( (precision_result * recall_result) / (precision_result + recall_result )))
        f1_score_data_4.append(f1_result)
    recall_data_4.append(recall_result)
    precision_data_4.append(precision_result)


# Visulization (Unnormalize)

# In[13]:


# Dataset 1

plt.scatter(K[:] , precision_data_1[:] , c='green', label='precision_veggies')
plt.scatter(K[:] , recall_data_1 [:] ,   c='blue', label='recall_veggies')
plt.scatter(K[:] , f1_score_data_1 [:] , c='red', label='f1_score_veggies')
plt.xlim([-1, 9])
plt.ylim([0,100])
plt.legend()
plt.xlabel('Total Number of Cluster')
plt.ylabel('Precision Recall F1_Score')
plt.title('Unnormalize Data Cluster Kmedian')
plt.show()


# Dataset 2

plt.scatter(K[:] , precision_data_2[:] , c='green', label='precision_animals')
plt.scatter(K[:] , recall_data_2 [:] ,   c='blue', label='recall_animals')
plt.scatter(K[:] , f1_score_data_2 [:] , c='red', label='f1_score_animals')
plt.xlim([-1, 9])
plt.ylim([0,100])
plt.legend()
plt.xlabel('Total Number of Cluster')
plt.ylabel('Precision Recall F1_Score')
plt.title('unnormalize Data Cluster Kmedian')
plt.show()


# Dataset 3

plt.scatter(K[:] , precision_data_3[:] , c='green', label='precision_fruits')
plt.scatter(K[:] , recall_data_3[:] ,   c='blue', label='recall_fruits')
plt.scatter(K[:] , f1_score_data_3[:] , c='red', label='f1_score_fruits')
plt.xlim([-1, 9])
plt.ylim([0,100])
plt.legend()
plt.xlabel('Total Number of Cluster')
plt.ylabel('Precision Recall F1_Score')
plt.title('unnormalize Data Cluster Kmedian')
plt.show()


# Dataset 4

plt.scatter(K[:] , precision_data_4[:] , c='green', label='precision_countries')
plt.scatter(K[:] , recall_data_4[:] ,   c='blue', label='recall_countries')
plt.scatter(K[:] , f1_score_data_4[:] , c='red', label='f1_score_countries')
plt.xlim([-1, 9])
plt.ylim([0,100])
plt.legend()
plt.xlabel('Total Number of Cluster')
plt.ylabel('Precision Recall F1_Score')
plt.title('unnormalize Data Cluster Kmedian')
plt.show()


# Normalize Dataset before apply the Cluster (Kmedian)

# In[14]:


# Dataset 1

minmaxscaler = MinMaxScaler()
normalize_data_1 = minmaxscaler.fit_transform(df_1)

km_data_1 = Kmedian(n_clusters=k, max_iter=100)
km_data_1.fit(normalize_data_1)
centroids_data_1 = km_data_1.centroids
predict_data_1 = km_data_1.predict(normalize_data_1)

# Dataset 2

normalize_data_2 = minmaxscaler.fit_transform(df_2)

km_data_2 = Kmedian(n_clusters=k, max_iter=100)
km_data_2.fit(normalize_data_2)
centroids_data_2 = km_data_2.centroids
predict_data_2   = km_data_2.predict(normalize_data_2)



# Dataset 3

normalize_data_3 = minmaxscaler.fit_transform(df_3)
km_data_3 = Kmedian(n_clusters=k, max_iter=100)
km_data_3.fit(normalize_data_3)
centroids_data_3 = km_data_3.centroids
predict_data_3   = km_data_3.predict(normalize_data_3)

# Dataset 4

normalize_data_4 = minmaxscaler.fit_transform(df_4)
km_data_4 = Kmedian(n_clusters=k, max_iter=100)
km_data_4.fit(normalize_data_4)
centroids_data_4 = km_data_4.centroids
predict_data_4   = km_data_4.predict(normalize_data_4)


# Kmedian Calculate (precision,recall,f1_score) (Normalization)

# In[15]:


# Dataset 1

precision_data_1 = []
recall_data_1    = []
f1_score_data_1  = []
for i in range(0,9):
    cluster_data = df_1[predict_data_1 == i , :]
    total_sum = 0
    max_sum   = 0
    for n in range(len(cluster_data)):
        total_sum = total_sum + np.sum(cluster_data[n])
        max_sum   = max_sum   + np.max(cluster_data[n])
    precision_result = int(abs((max_sum/total_sum) * 100))
    recall_result    = int(abs((max_sum/(total_sum+9)) * 100))
    if ((precision_result+recall_result) == 0):
        f1_score_data_1.append(0)
    else:
        f1_result        = int(2 * ( (precision_result * recall_result) / (precision_result + recall_result )))
        f1_score_data_1.append(f1_result)
    recall_data_1.append(recall_result)
    precision_data_1.append(precision_result)


# Dataset 2

precision_data_2 = []
recall_data_2    = []
f1_score_data_2  = []
for i in range(0,9):
    cluster_data = df_2[predict_data_2 == i , :]
    total_sum = 0
    max_sum   = 0
    for n in range(len(cluster_data)):
        total_sum = total_sum + np.sum(cluster_data[n])
        max_sum   = max_sum   + np.max(cluster_data[n])
    precision_result = int(abs((max_sum/total_sum) *100 ))
    recall_result    = int(abs((max_sum/(total_sum+9)) * 100))
    if ((precision_result+recall_result) == 0):
        f1_score_data_2.append(0)
    else:
        f1_result        = int(2 * ( (precision_result * recall_result) / (precision_result + recall_result )))
        f1_score_data_2.append(f1_result)
    recall_data_2.append(recall_result)
    precision_data_2.append(precision_result)
    

# Dataset 3

precision_data_3 = []
recall_data_3    = []
f1_score_data_3  = []
for i in range(0,9):
    cluster_data = df_2[predict_data_3 == i , :]
    total_sum = 0
    max_sum   = 0
    for n in range(len(cluster_data)):
        total_sum = total_sum + np.sum(cluster_data[n])
        max_sum   = max_sum   + np.max(cluster_data[n])
    precision_result = int(abs((max_sum/total_sum) *100 ))
    recall_result    = int(abs((max_sum/(total_sum+9)) * 100))
    if ((precision_result+recall_result) == 0):
        f1_score_data_3.append(0)
    else:
        f1_result        = int(2 * ( (precision_result * recall_result) / (precision_result + recall_result )))
        f1_score_data_3.append(f1_result)
    recall_data_3.append(recall_result)
    precision_data_3.append(precision_result)

    
# Dataset 4

precision_data_4 = []
recall_data_4    = []
f1_score_data_4  = []
for i in range(0,9):
    cluster_data = df_2[predict_data_4 == i , :]
    total_sum = 0
    max_sum   = 0
    for n in range(len(cluster_data)):
        total_sum = total_sum + np.sum(cluster_data[n])
        max_sum   = max_sum   + np.max(cluster_data[n])
    precision_result = int(abs((max_sum/total_sum) *100 ))
    recall_result    = int(abs((max_sum/(total_sum+9)) * 100))
    if ((precision_result+recall_result) == 0):
        f1_score_data_4.append(0)
    else:
        f1_result        = int(2 * ( (precision_result * recall_result) / (precision_result + recall_result )))
        f1_score_data_4.append(f1_result)
    recall_data_4.append(recall_result)
    precision_data_4.append(precision_result)


# Visulization (Normalize)

# In[16]:


# Dataset 1

plt.scatter(K[:] , precision_data_1[:] , c='green', label='precision_veggies')
plt.scatter(K[:] , recall_data_1 [:] ,   c='blue', label='recall_veggies')
plt.scatter(K[:] , f1_score_data_1 [:] , c='red', label='f1_score_veggies')
plt.xlim([-1, 9])
plt.ylim([0,100])
plt.legend()
plt.xlabel('Total Number of Cluster')
plt.ylabel('Precision Recall F1_Score')
plt.title('Normalize Data Cluster Kmedian')
plt.show()


# Dataset 2

plt.scatter(K[:] , precision_data_2[:] , c='green', label='precision_animals')
plt.scatter(K[:] , recall_data_2 [:] ,   c='blue', label='recall_animals')
plt.scatter(K[:] , f1_score_data_2 [:] , c='red', label='f1_score_animals')
plt.xlim([-1, 9])
plt.ylim([0,100])
plt.legend()
plt.xlabel('Total Number of Cluster')
plt.ylabel('Precision Recall F1_Score')
plt.title('Normalize Data Cluster Kmedian')
plt.show()


# Dataset 3

plt.scatter(K[:] , precision_data_3[:] , c='green', label='precision_fruits')
plt.scatter(K[:] , recall_data_3[:] ,   c='blue', label='recall_fruits')
plt.scatter(K[:] , f1_score_data_3[:] , c='red', label='f1_score_fruits')
plt.xlim([-1, 9])
plt.ylim([0,100])
plt.legend()
plt.xlabel('Total Number of Cluster')
plt.ylabel('Precision Recall F1_Score')
plt.title('Normalize Data Cluster Kmedian')
plt.show()


# Dataset 4

plt.scatter(K[:] , precision_data_4[:] , c='green', label='precision_countries')
plt.scatter(K[:] , recall_data_4[:] ,   c='blue', label='recall_countries')
plt.scatter(K[:] , f1_score_data_4[:] , c='red', label='f1_score_countries')
plt.xlim([-1, 9])
plt.ylim([0,100])
plt.legend()
plt.xlabel('Total Number of Cluster')
plt.ylabel('Precision Recall F1_Score')
plt.title('Normalize Data Cluster Kmedian')
plt.show()


# In[ ]:





# In[ ]:




