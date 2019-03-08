import numpy as np
import Helper as hp


# imageId to Index
IdsIdxMap = hp.load("IdsIdxMap.txt")

# Index to Image Id
IdxIdsMap = hp.load("IdxIdsMap.txt")

k =input("Enter no. outgoing edges k :  ")


b = hp.load("ImageImageSimilarityMatrix" + k +".txt")

#adjecenty matrix
adj_matrix = np.zeros((len(IdxIdsMap), len(IdxIdsMap)))

for i in range(0, len(adj_matrix)):
    q =b[IdxIdsMap[i]];
    for value in q:
        adj_matrix[i][IdsIdxMap[value]] = 1.0

hp.save(adj_matrix, "adj_matrix.txt")

np.savetxt("adj_matrix_hR.csv", adj_matrix, delimiter=",")
