import pandas as pd
import numpy as np
from sklearn import neighbors
import matplotlib.pyplot as plt

offline_rss = pd.read_excel("../data/offline_rss.xlsx", header=None).values
offline_location = pd.read_excel("../data/offline_location.xlsx", header=None).values
trace_rss = pd.read_excel("../data/trace_rss.xlsx", header=None).values
trace_location = pd.read_excel("../data/trace_location.xlsx", header=None).values

def accuracy(predictions, labels):
    return np.mean(np.sqrt(np.sum((predictions - labels)**2, 1)))

# knn regression
def knn_reg(k):
	knn_reg = neighbors.KNeighborsRegressor(n_neighbors=k, weights='uniform', metric='euclidean')
	predict = knn_reg.fit(offline_rss, offline_location).predict(trace_rss)
	acc = accuracy(predict, trace_location)
	return predict, acc

if __name__ == "__main__":
    k = 3
    predict, accuracy = knn_reg(k)
    plt.scatter(trace_location[:,0], trace_location[:,1], c="green")
    # plot函数返回一个列表，加逗号拆分出了第一个元素
    l1, = plt.plot(trace_location[:,0], trace_location[:,1], c="blue")
    plt.scatter(predict[:,0], predict[:,1], c="red")
    l2, = plt.plot(predict[:,0], predict[:,1], c="orange")

    plt.legend(handles=[l1,l2],labels=['real tracks','predicting'],loc='best')
    plt.grid()
    print("k:", k, ", accuracy: ", round(accuracy, 3), "m")
    plt.show()