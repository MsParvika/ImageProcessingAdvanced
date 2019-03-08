import helpers as hp;
import SpareMatrixPR as SMPR;
import ShowImages as si;
import configparser as cp
import os

config = cp.ConfigParser()
config.read(os.getcwd()+'/config.ini')

k =input("Enter no. dominant images :  ")

# adjacency Matrix saved as part of task 1
adj_matrix = hp.load("adj_matrix.txt")

# imageId to Index
IdsIdxMap = hp.load("IdsIdxMap.txt")

# Index to Image Id
IdxIdsMap = hp.load("IdxIdsMap.txt")


pr2=SMPR.pagerank_scipy(adj_matrix, alpha=0.85, tol=1.0e-10)

imageidx =  pr2.argsort()[-int(k):][::-1]

imageidxList = []

for idx in imageidx:
    imageidxList.append(IdxIdsMap[idx])


devset_path = config['TASK3']['devset_path']
si.fetchImages(imageidxList, devset_path)


