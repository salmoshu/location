import pandas as pd
import numpy as np
from sklearn import neighbors

offline_rss = pd.read_excel("./Han/Data/offline_rss.xlsx").values
offline_location = pd.read_excel("./Han/Data/offline_location.xlsx").values
online_rss = pd.read_excel("./Han/Data/online_rss.xlsx").values
online_location = pd.read_excel("./Han/Data/online_location.xlsx").values

def accuracy(predictions, labels):
    return np.mean(np.sqrt(np.sum((predictions - labels)**2, 1)))

# knn regression
def knn_reg(k):
	knn_reg = neighbors.KNeighborsRegressor(n_neighbors=k, weights='uniform', metric='euclidean')
	predictions = knn_reg.fit(offline_rss, offline_location).predict(online_rss)
	acc = accuracy(predictions, online_location)
	print("k:", k, ", accuracy: ", round(acc, 3), "m")

if __name__ == "__main__":
	for i in range(0, 5):
		knn_reg(i+1)