###

#Python implementaion of A Linear Time-Complexity k-Means Algorithm Using Cluster Shifting with Kmean++ initialization

###
import pandas as pd
import numpy as np
import random

class KMeans(object):
    def __init__(self,k=2,epochs=100,error=0.001,alpha = 10e-6):
        self.k = k
        self.epochs = epochs
        self.error =error
        self.centroids = []
        self.alpha = alpha
        self.sum_of_list_of_update_vectors = [0] * k
        
    def distance(self,centroid,point):
        n = len(centroid)
        return (np.sum(centroid - point)**n)**(1/n)
    
    def squared_distance(self,centroid,point):
        return np.sqrt(np.sum((centroid - point)**2))
    
    def find_centroids(self,data):
        if len(self.centroids)==0:
            self.centroids.append(data[random.randint(0,len(data))])
        max_dist_point = None
        max_dist = 0
        for d in data:
            for centroid in self.centroids:
                dis = self.squared_distance(centroid,d)
                if dis > max_dist:
                    max_dist = dis
                    max_dist_point = d
        self.centroids.append(max_dist_point)
        if len(self.centroids) < self.k:
            self.find_centroids(data)
            
            
    def fit(self,data):
        clusters = {}
        shape = data.shape
        self.centroids = []
        
        #Get the centroids according to kmeans++ algorithm
        self.find_centroids(data)
        
        for i in range(self.k):
            self.sum_of_list_of_update_vectors[i] = 0
        for epoch in range(self.epochs):
            
            # Resetting Clusters
            for i in range(self.k):
                clusters[i] = []
            
            # Get the nearest data points to the centroid
            for d in data:
                distances = [self.squared_distance(centroid,d) for centroid in self.centroids]
                cluster_index = distances.index(min(distances))
                clusters[cluster_index].append(d)
                
                
            global_centroid = np.array([np.mean(data[:,i]) for i in range(data.shape[1])])
            
            # Calculating direction vectors and update vector
            
            direction_vector = self.centroids - global_centroid
            update_vector = self.alpha * direction_vector
            temp_arr = np.array([])
            for i in range(self.k):
                
                self.sum_of_list_of_update_vectors[i] += update_vector[i]
                updated_data_points = np.array(clusters[i]) + update_vector[i]
                self.centroids[i] += update_vector[i]
                temp_arr = np.append(temp_arr,updated_data_points)
            data = temp_arr.reshape(shape)
            
            # update the centroids
            previous_centroid = self.centroids.copy()
            for i in range(self.k):
                self.centroids[i] = np.array(clusters[i]).mean(axis=0)
            
            error = np.sum(np.array(previous_centroid) - np.array(self.centroids))
            print("Epoch "+str(epoch)+" Loss = "+str(abs(error)))
            if abs(error) < self.error:
                break
    #Returning the cluster classes for each data
    def predict(self,data):
        prediction = []
        for d in data:
            distances = [self.distance(self.centroids[index],d+self.sum_of_list_of_update_vectors[index]) for index in range(len(self.centroids))]
            prediction.append(distances.index(min(distances)))
        return np.array(prediction)
