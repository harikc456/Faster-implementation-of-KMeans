###

# Python implementation of Normal Kmeans with kmeans++ cluster initializing

###
import pandas as pd
import numpy as np
import random

class KMeans(object):
    def __init__(self,k=2,epochs=100,error=0.001):
        self.k = k
        self.epochs = epochs
        self.error =error
        self.centroids = []
        
    def distance(self,centroid,point):
        n = len(centroid)
        return np.sum((centroid - point)**n)**(1/n)
    
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
        self.centroids = []
        
        #Get the centroids according to kmeans++ algorithm
        self.find_centroids(data)
        
        for epoch in range(self.epochs):
            
            for i in range(self.k):
                clusters[i] = []
            
            # Get the nearest data points to the centroid
            for d in data:
                distances = [self.squared_distance(centroid,d) for centroid in self.centroids]
                cluster_index = distances.index(min(distances))
                clusters[cluster_index].append(d)
            # update the centroids
            previous_centroid = self.centroids.copy()
            for i in range(self.k):
                self.centroids[i] = np.array(clusters[i]).mean(axis=0)
            error = np.sum(np.array(previous_centroid) - np.array(self.centroids))
            print("Epoch "+str(epoch)+" Loss = "+str(abs(error)))
            if abs(error) < self.error:
                break
    def predict(self,data):
        preds = []
        for d in data:
            distances = [self.distance(centroid,d) for centroid in self.centroids]
            cluster_index = distances.index(min(distances))
            preds.append(cluster_index)
        return np.array(preds)
