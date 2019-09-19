# Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


def elbow_method(X,num_clusters = 10,method = 'k-means++'):
	"""
	One method to validate the number of clusters is the 'Elbow method'. The idea of the elbow method 
	is to run k-means clustering on the dataset for a range of values of k and for each value of k 
	calculate the sum of squared errors (SSE). Then, plot a line chart of the SSE for each value of k. 
	If the line chart looks like an arm, then the "elbow" on the arm is the value of k that is the best. 

	Inputs:
  - X : Training instances to cluster. It must be noted that the data will be converted to C ordering, 
  which will cause a memory copy if the given data is not C-contiguous.
	- num_clusters : number of clusters to test
	- n_clusters : The number of clusters to form as well as the number of centroids to generate
	- init (default = ‘k-means++’) : Method for initialization
	- n_init (default = 10) : Number of time the k-means algorithm will be run with different centroid seeds. 
	The final results will be the best output of n_init consecutive runs in terms of inertia
	- max_iter (default = 300) : Maximum number of iterations of the k-means algorithm for a single run
	- random_state = Determines random number generation for centroid initialization. Use an int to make the randomness deterministic

	Outputs:
	This function will plot a figure that looks like an arm 
	"""
  
  # Generating the results of each application of the KMeans algorithm with the chosen num_clusters
	wcss =[]
	for i in range(1,int(num_clusters+1)):
	    kmeans = KMeans(n_clusters=i,init=method,max_iter=300,n_init=10,random_state=0)
	    kmeans.fit(X)
	    wcss.append(kmeans.inertia_)
  
  # plotting the 'Elbow plot'
	plt.plot(range(1,int(num_clusters+1)),wcss)
	plt.title('The elbow method')
	plt.xlabel('Number of clusters')
	plt.ylabel('WCSS')
	plt.show()
