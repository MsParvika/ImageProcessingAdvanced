from src import Helper as hp;
from src import SpareMatrixPR as SMPR;
from src import ShowImages as si;
import time;
import configparser as cp;
import os;



config = cp.ConfigParser()
config.read(os.getcwd()+'/config.ini')

k =input("Enter no. dominant images :  ");

n_k = int(config['TASK4']['n_k']);

# adjacency Matrix saved as part of task 1
adj_matrix = hp.load("adj_matrix.txt")

# imageId to Index
IdsIdxMap = hp.load("IdsIdxMap.txt");

# Index to Image Id
IdxIdsMap = hp.load("IdxIdsMap.txt");

arr = [];
arr_ind = []
print("Enter the image id ")

for i in range(0,n_k,1):
    arr.append(input(""))
    print (arr[i])
    arr_ind.append(IdsIdxMap[arr[i]])

print(arr_ind);

tic = time.clock();

pr2=SMPR.pagerank_scipy(adj_matrix, alpha=0.85, tol=1.0e-10, personalised=True, indxs=arr_ind);

imageidx =  pr2.argsort()[-int(k):][::-1];

print(pr2);
print(len(pr2));
print(imageidx);
print(pr2[imageidx]);

imageidxList = [];

for idx in imageidx:
    imageidxList.append(IdxIdsMap[idx]);

toc = time.clock();

print(toc-tic);

devset_path = config['TASK4']['devset_path']
si.fetchImages(imageidxList, devset_path);