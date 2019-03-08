from collections import OrderedDict;
import helpers as hp;
import configparser as cp
import os
import scipy.sparse
import numpy as np
from scipy.linalg import pinv
from scipy.sparse import csr_matrix


config = cp.ConfigParser();
config.read(os.getcwd()+'/config.ini')
# degree of freedom
k = int(config['PREPROCESSING']['k'])
run_task_2 = bool(int(config['PREPROCESSING']['run_task_2']))

# imageId to Index
IdsIdxMap = {}

# Index to Image Id
IdxIdsMap = {}

ImageWordTfIdfMap = {}

def readFromCSVTD(fileName):
    with open(fileName, "r", encoding="utf8") as ins:
        i = 0
        w = 1
        for line in ins:
            row = line.split(' ')
            wordTfIdfMap = {};
            for j in range(len(row)):
                if(j==0):
                    IdsIdxMap[row[j]]=i
                    IdxIdsMap[i]=row[j]
                    ImageWordTfIdfMap[row[j]]=wordTfIdfMap
                    continue
                if(row[j].find('"')!= -1):
                    key = row[j].replace('"', '')
                    wordTfIdfMap[key] = float(row[j+3])
            i = i +1
    return ImageWordTfIdfMap

image_file = config['PREPROCESSING']['raw_image_path']
readFromCSVTD(image_file);

hp.save(IdsIdxMap, "IdsIdxMap.txt")
IdsIdxMap = {};
hp.save(IdxIdsMap, "IdxIdsMap.txt")
IdxIdsMap = {};

cosineBoolean = {}


def findCosineSimilarity(key1, map1, key2, map2):

    if len(map1) > len(map2):
        tempKey = key1
        key1 = key2
        key2 = tempKey
        tempMap = map1
        map1 = map2
        map2 = tempMap
    try:
        return cosineBoolean[key1+key2]
    except KeyError:
        dot_prod = 0;
        for key, value in map1.items():
            prod = 0;
            try:
                value1 = map2[key]
            except KeyError:
                continue
            prod = value1 * value
            dot_prod = dot_prod + prod
        cosineBoolean[key1 + key2] = dot_prod / (len(map1) + len(map2))
        return cosineBoolean[key1 + key2]

ImageImageSimilarityMatrix = {}

if not run_task_2:
    i = 0;
    for key, value in ImageWordTfIdfMap.items():
        if i % 20 == 0:
            print(i)
        try:
            q = ImageImageSimilarityMatrix[key]
        except KeyError:
            q = OrderedDict();
        ImageImageSimilarityMatrix[key] = q;
        for key1, value1 in ImageWordTfIdfMap.items():
            try:
                q1 = ImageImageSimilarityMatrix[key1]
            except KeyError:
                q1 = OrderedDict();
            ImageImageSimilarityMatrix[key1] = q1
            csValue = findCosineSimilarity(key, value, key1, value1);

            if (key1 in q):
                pass
            else:
                q[key1] = csValue;

            if (key in q1):
                pass
            else:
                q1[key] = csValue;

            q = OrderedDict(sorted(q.items(), key=lambda kv: kv[1], reverse=True))
            q1 = OrderedDict(sorted(q1.items(), key=lambda kv: kv[1], reverse=True))
            if(len(q) > k ):
                for x in list(reversed(list(q))):
                    q.pop(x)
                    break
            if (len(q1) > k):
                for x in list(reversed(list(q1))):
                    q1.pop(x)
                    break
            ImageImageSimilarityMatrix[key] = q
            ImageImageSimilarityMatrix[key1] = q1
        i = i + 1
    cosineBoolean = {}

    hp.save(ImageImageSimilarityMatrix, "ImageImageSimilarityMatrix"+ str(k) +".txt")

def pre_process_task2():

    # imageId to Index
    IdsIdxMap = hp.load("IdsIdxMap.txt")

    # Index to Image Id
    IdxIdsMap = hp.load("IdxIdsMap.txt")
    b = hp.load("ImageImageSimilarityMatrix"+ str(k) +".txt")

    #adjecenty matrix
    adj_matrix = np.zeros((len(IdxIdsMap), len(IdxIdsMap)))
    weighted_adj_matrix = np.zeros((len(IdxIdsMap), len(IdxIdsMap)))

    for i in range(0, len(adj_matrix)):
        q = b[IdxIdsMap[i]]
        for value in q:
            adj_matrix[i][IdsIdxMap[value]] = 1.0;
            weighted_adj_matrix[i][IdsIdxMap[value]] = q[value]

    A = np.array(adj_matrix)
    W = np.array(weighted_adj_matrix)

    D_out = np.eye(A.shape[0]) * np.sum(A, axis=1, keepdims=True)
    D_out = np.sqrt(pinv(D_out))
    D_out = scipy.sparse.csr_matrix(D_out)

    D_in = np.eye(A.shape[0]) * np.sum(A, axis=0, keepdims=True)
    D_in = np.sqrt(pinv(D_in))
    D_in = scipy.sparse.csr_matrix(D_in)

    B = csr_matrix.dot(csr_matrix.dot(csr_matrix.dot(csr_matrix.dot(D_out,A),D_in),A.T),D_out)

    C = csr_matrix.dot(csr_matrix.dot(csr_matrix.dot(csr_matrix.dot(D_in,A.T),D_out),A),D_in)

    sparse_A_matrix = scipy.sparse.csr_matrix(A)
    scipy.sparse.save_npz(os.getcwd()+'/data/A_'+str(k)+'.npz', sparse_A_matrix)

    sparse_W_matrix = scipy.sparse.csr_matrix(W)
    scipy.sparse.save_npz(os.getcwd()+'/data/W_'+str(k)+'.npz', sparse_W_matrix)

    sparse_B_matrix = scipy.sparse.csr_matrix(B)
    scipy.sparse.save_npz(os.getcwd()+'/data/B_'+str(k)+'.npz', sparse_B_matrix)

    sparse_C_matrix = scipy.sparse.csr_matrix(C)
    scipy.sparse.save_npz(os.getcwd()+'/data/C_'+str(k)+'.npz', sparse_C_matrix)

if run_task_2:
    pre_process_task2()
