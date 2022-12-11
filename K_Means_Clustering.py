import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def Calculate_New_Centroids(x_data, total_clusters, prev_centroids):
    clusters= [[] for i in range(total_clusters)]  
    
    for idx, point in enumerate(x_data):
        nearest_centroid= np.argmin(np.sqrt(np.sum((prev_centroids-point)**2, axis=1)))
        clusters[nearest_centroid].append(idx)
        
    new_centroids=np.zeros((total_clusters, len(x_data[0])))
    for idx, cluster_points in enumerate(clusters):
        new_centroids[idx]= np.mean(x_data[cluster_points], axis=0)
    
    return  (clusters,new_centroids)

def plot_clusters(x_data, clusters):
    colors=['red', 'green', 'blue', 'black', 'orange', 'pink']
    
    for idx, cluster_point in enumerate(clusters): 
        for point in cluster_point:
            plt.scatter(x_data[point, 0], x_data[point, 2], color= colors[idx])
    
    plt.show()
    
def KMeansClustering(x_data, total_clusters, Max_iterations= 100):
    centroids= np.zeros((total_clusters, len(x_data[0])))
    
    for i in range(total_clusters):
        centroids[i]= x_data[i]
    centroids= np.array(centroids)
    
    for j in range(Max_iterations):
        clusters, new_centroids= Calculate_New_Centroids(x_data, total_clusters, centroids)
        
        diff= np.sum(np.sqrt(np.sum((new_centroids-centroids)**2, axis=1)))
        
        if diff==0:
            break
        centroids= new_centroids
    
    plot_clusters(x_data, clusters)
    return centroids

iris= load_iris()

x= iris.data
print("\n\nThe x_dataset: \n",x)

print(x.shape)
clusters_count=5
cluster_centroid= KMeansClustering(x, clusters_count)

for idx, point in enumerate(cluster_centroid):
    print("Centroid of cluster ", idx)
    print(": ", point);
    print("\n")