# Import relevant libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Open using pandas

animals = pd.read_csv('animals.csv', delimiter = ' ',header = None)
countries = pd.read_csv('countries.csv', delimiter = ' ' ,header = None)
fruits = pd.read_csv('fruits.csv', delimiter = ' ' ,header = None)
veggies = pd.read_csv('veggies.csv', delimiter = ' ' ,header = None)
train_dataset  = np.concatenate([animals, countries, fruits, veggies])
X = np.delete(train_dataset,0,axis=1)
animals['newCol'] = animals.apply(lambda x: 0, axis=1)
countries['newCol'] = countries.apply(lambda x: 1, axis=1)
fruits['newCol'] = fruits.apply(lambda x: 2, axis=1)
veggies['newCol'] = veggies.apply(lambda x: 3, axis=1)
train_dataset_1  = np.concatenate([animals, countries, fruits, veggies])
K_list=[]
for k in range(1,10):
    K_list.append(k)
td=[]
td1=[]
td2=[]
total_animal=len(animals.index)
total_countries=len(countries)
total_fruits=len(fruits)
total_veggies=len(veggies)

# K means
for k in range(1,10):
    error_rate=2
    sum=0
    sum1=0
    sum2=0
    clusters={}
    sp={}
    samples, features = X.shape
    np.random.seed(40)
    centroids = []
    random_samples = np.random.choice(samples, k, replace = False)
    for r in random_samples:
        centroids.append(X[r])
    while(error_rate!=0):
        for i in range(k):
            clusters[i] = []
            sp[i]=[]
        for data in X:
            euc_dist = []
            for j in range(k):
                euc_dist.append(np.linalg.norm(data - centroids[j]))
            clusters[euc_dist.index(min(euc_dist))].append(data)
        centroids_old = centroids
        centroids = np.zeros((k, features))
        for ind, clust in clusters.items():
            mean_clust = np.mean(clusters[ind], axis=0)
            centroids[ind] = mean_clust
        error_rate = np.linalg.norm(centroids - centroids_old)
    for x in train_dataset_1:
        t=x[1:301]
        for i in range(0,k):
            for j in range(0,len(clusters[i])):
                if(np.array_equal(t,clusters[i][j])==True):
                    sp[i].append(x[301])
    for key,value in sp.items():
        total_count=0
        c_animal=0
        c_countries=0
        c_fruits=0
        c_veggies=0  
        for i in range(len(value)):
            if(value[i]==0):
                c_animal=c_animal+1
                total_count+=1
            if(value[i]==1):
                c_countries=c_countries+1
                total_count+=1
            if(value[i]==2):
                c_fruits=c_fruits+1
                total_count+=1
            if(value[i]==3):
                c_veggies=c_veggies+1
                total_count+=1
        sum=sum+(c_animal*c_animal/total_count)+(c_countries*c_countries/total_count)+(c_fruits*c_fruits/total_count)+(c_veggies*c_veggies/total_count)
        sum1=sum1+(c_animal*c_animal/total_animal)+(c_countries*c_countries/total_countries)+(c_fruits*c_fruits/total_fruits)+(c_veggies*c_veggies/total_veggies)
    sum2=sum2+(2*sum*sum1)/(sum+sum1)
    td.append(sum/327)
    td1.append(sum1/327)
    td2.append(sum2/327)
plt.plot(K_list, td, label="Precision")
plt.plot(K_list, td1, label="Recall")
plt.plot(K_list, td2, label="Fscore")
plt.xlabel('Number of Clusters')
plt.ylabel("Score")
plt.legend()
plt.show()

#median
td=[]
td1=[]
td2=[]
for k in range(1,10):
    error_rate=2
    sum=0
    sum1=0
    sum2=0
    clusters={}
    sp={}
    samples, features = X.shape
    np.random.seed(40)
    centroids = []
    random_samples = np.random.choice(samples, k, replace = False)
    for r in random_samples:
        centroids.append(X[r])
    while(error_rate!=0):
        for i in range(k):
            clusters[i] = []
            sp[i]=[]
        for data in X:
            euc_dist = []
            for j in range(k):
                euc_dist.append(np.linalg.norm(data - centroids[j],ord=1))
            clusters[euc_dist.index(min(euc_dist))].append(data)
        centroids_old = centroids
        centroids = np.zeros((k, features))
        for ind, clust in clusters.items():
            median_clust = np.median(clusters[ind], axis=0)
            centroids[ind] = median_clust
        error_rate = np.linalg.norm(centroids - centroids_old)
        for x in train_dataset_1:
            t=x[1:301]
            for i in range(0,k):
                for j in range(0,len(clusters[i])):
                    if(np.array_equal(t,clusters[i][j])==True):
                        sp[i].append(x[301])
    for key,value in sp.items():
        total_count=0
        count_animal=0
        count_countries=0
        count_fruits=0
        count_veggies=0  
        for i in range(len(value)):
            if(value[i]==0):
                count_animal=count_animal+1
                total_count+=1
            if(value[i]==1):
                count_countries=count_countries+1
                total_count+=1
            if(value[i]==2):
                count_fruits=count_fruits+1
                total_count+=1
            if(value[i]==3):
                count_veggies=count_veggies+1
                total_count+=1
        sum=sum+(count_animal*count_animal/total_count)+(count_countries*count_countries/total_count)+(count_fruits*count_fruits/total_count)+(count_veggies*count_veggies/total_count)
        sum1=sum1+(count_animal*count_animal/total_animal)+(count_countries*count_countries/total_countries)+(count_fruits*count_fruits/total_fruits)+(count_veggies*count_veggies/total_veggies)
    sum2=sum2+(2*sum*sum1)/(sum+sum1)
    td.append(sum/327)
    td1.append(sum1/327)
    td2.append(sum2/327)
print("hereeee")
plt.plot(K_list, td, label="Precision")
plt.plot(K_list, td1, label="Recall")
plt.plot(K_list, td2, label="Fscore")
plt.xlabel('Number of Clusters')
plt.ylabel("Score")   
plt.legend()
plt.show() 
X=train_dataset_1[:,1:301]
X_Normalised=X/np.linalg.norm(X)
e=[]
for i in range(50):
    e.append(np.append(X_Normalised[i],0))
for i in range(50,211):
    e.append(np.append(X_Normalised[i],1))
for i in range(211,269):
    e.append(np.append(X_Normalised[i],2))
for i in range(269,327):
    e.append(np.append(X_Normalised[i],3))

td=[]
td1=[]
td2=[]
#  Kmeans with L2
for k in range(1,10):
    sum=0
    sum1=0
    sum2=0
    error_rate=2
    sp={}
    clusters={}
    cluster_dict={}
    samples, features = X_Normalised.shape
    np.random.seed(40)
    centroids = []
    random_samples = np.random.choice(samples, k, replace = False)
    for r in random_samples:
        centroids.append(X_Normalised[r])
    while(error_rate!=0):
        for i in range(k):
            clusters[i] = []
            sp[i]=[]
        for data in X_Normalised:
            euclidean = []
            for j in range(k):
                euclidean.append(np.linalg.norm(data - centroids[j]))
                
            clusters[euclidean.index(min(euclidean))].append(data)
        centroids_old = centroids
        centroids = np.zeros((k, features))
        for ind in clusters.keys():
            mean = np.mean(clusters[ind], axis=0)
            centroids[ind] = mean
        error_rate = np.linalg.norm(centroids - centroids_old)
        for x in e:
            t=x[0:300]
            for i in range(0,k):
                for j in range(0,len(clusters[i])):
                    if(np.array_equal(t,clusters[i][j])==True):
                        sp[i].append(x[300])
    for key,value in sp.items():
        total_count=0
        count_animal=0
        count_countries=0
        count_fruits=0
        count_veggies=0  
        for i in range(len(value)):
            if(value[i]==0):
                count_animal=count_animal+1
                total_count+=1
            if(value[i]==1):
                count_countries=count_countries+1
                total_count+=1
            if(value[i]==2):
                count_fruits=count_fruits+1
                total_count+=1
            if(value[i]==3):
                count_veggies=count_veggies+1
                total_count+=1
        sum=sum+(count_animal*count_animal/total_count)+(count_countries*count_countries/total_count)+(count_fruits*count_fruits/total_count)+(count_veggies*count_veggies/total_count)
        sum1=sum1+(count_animal*count_animal/total_animal)+(count_countries*count_countries/total_countries)+(count_fruits*count_fruits/total_fruits)+(count_veggies*count_veggies/total_veggies)
    sum2=sum2+(2*sum*sum1)/(sum+sum1)
    td.append(sum/327)
    td1.append(sum1/327)
    td2.append(sum2/327)
print("hereeee")
plt.plot(K_list, td, label="Precision")
plt.plot(K_list, td1, label="Recall")
plt.plot(K_list, td2, label="Fscore")
plt.xlabel('Number of Clusters')
plt.ylabel("Score")
plt.legend()  
plt.show() 

#  Kmedians with L2
td=[]
td1=[]
td2=[]
for k in range(1,10):
    sum=0
    sum1=0
    sum2=0
    error_rate=2
    sp={}
    clusters={}
    cluster_dict={}
    samples, features = X_Normalised.shape
    np.random.seed(40)
    centroids = []
    random_samples = np.random.choice(samples, k, replace = False)
    for r in random_samples:
        centroids.append(X_Normalised[r])
    while(error_rate!=0):
        for i in range(k):
            clusters[i] = []
            sp[i]=[]
        for data in X_Normalised:
            euclidean = []
            for j in range(k):
                euclidean.append(np.linalg.norm(data - centroids[j],ord=1))            
            clusters[euclidean.index(min(euclidean))].append(data)
        centroids_old = centroids
        centroids = np.zeros((k, features))
        for ind in clusters.keys():
            median = np.median(clusters[ind], axis=0)
            centroids[ind] = median
        error_rate = np.linalg.norm(centroids - centroids_old)
        for x in e:
            t=x[0:300]
            for i in range(0,k):
                for j in range(0,len(clusters[i])):
                    if(np.array_equal(t,clusters[i][j])==True):
                        sp[i].append(x[300])
    for key,value in sp.items():
        total_count=0
        count_animal=0
        count_countries=0
        count_fruits=0
        count_veggies=0  
        for i in range(len(value)):
            if(value[i]==0):
                count_animal=count_animal+1
                total_count+=1
            if(value[i]==1):
                count_countries=count_countries+1
                total_count+=1
            if(value[i]==2):
                count_fruits=count_fruits+1
                total_count+=1
            if(value[i]==3):
                count_veggies=count_veggies+1
                total_count+=1
        sum=sum+(count_animal*count_animal/total_count)+(count_countries*count_countries/total_count)+(count_fruits*count_fruits/total_count)+(count_veggies*count_veggies/total_count)
        sum1=sum1+(count_animal*count_animal/total_animal)+(count_countries*count_countries/total_countries)+(count_fruits*count_fruits/total_fruits)+(count_veggies*count_veggies/total_veggies)
    sum2=sum2+(2*sum*sum1)/(sum+sum1)
    td.append(sum/327)
    td1.append(sum1/327)
    td2.append(sum2/327)
print("hereeee")
plt.plot(K_list, td, label="Precision") 
plt.plot(K_list, td1, label="Recall")
plt.plot(K_list, td2, label="Fscore")
plt.xlabel('Number of Clusters')
plt.ylabel("Score")   
plt.legend()  
plt.show() 
