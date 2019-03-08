import helpers
import numpy as np
import scipy
import scipy.sparse
import sys
import math
import configparser as cp
import Visualize as vis
import os

def findclusters(U,cluster_center_vectors):
    legit = 0
    cdict = [[] for i in range(len(cluster_center_vectors))]
    no_of_nodes = U.shape[0]
    for ii in range(no_of_nodes):
        mxsim = 0
        cluster_c = np.random.randint(len(cluster_center_vectors))
        for jj in range(len(cluster_center_vectors)):
            dp = np.sum(U[ii]*cluster_center_vectors[jj])
            m1 = np.sum(U[ii]*U[ii])
            m2 = np.sum(cluster_center_vectors[jj]*cluster_center_vectors[jj])
            tl = dp/(np.sqrt(m1*m2)+math.pow(10,-6))
            if tl>mxsim:
                mxsim = tl
                cluster_c = jj
        cdict[cluster_c].append(ii)
    return cdict

def updatecenters(U,cdict):
    new_cluster_centers = []
    for cd in cdict:
        ncenter = np.full([U.shape[1]],0.0)
        for vv in cd:
            ncenter += U[vv]
        ncenter /= len(cd)
        new_cluster_centers.append(ncenter)
    return new_cluster_centers
        
def sort_closest_first(U,clist,cluster_center_vectors):
    rlist = []
    for ii in range(len(cluster_center_vectors)):
        cscore = []
        for jj in range(len(clist[ii])):
            dp = np.sum(U[clist[ii][jj]]*cluster_center_vectors[ii])
            m1 = np.sum(U[clist[ii][jj]]*U[clist[ii][jj]])
            m2 = np.sum(cluster_center_vectors[ii]*cluster_center_vectors[ii])
            tl = dp/(np.sqrt(m1*m2)+math.pow(10,-6))
            cscore.append(tl)
        rt = np.flip(np.argsort(cscore))
        for jj in range(len(rt)):
        	t = clist[ii][rt[jj]]
        	rt[jj] = t
        rlist.append(rt)
    return rlist


def main():
    ip = input('Please enter the nearest neighbors k -- ')

    if sys.argv[1]=='U':
        if ip=='5':
            B = (scipy.sparse.load_npz('./data/B_5.npz')).todense()
            C = (scipy.sparse.load_npz('./data/C_5.npz')).todense()
        elif ip == '3':
            B = (scipy.sparse.load_npz('./data/B_3.npz')).todense()
            C = (scipy.sparse.load_npz('./data/C_3.npz')).todense()
        else:
            B = (scipy.sparse.load_npz('./data/B_10.npz')).todense()
            C = (scipy.sparse.load_npz('./data/C_10.npz')).todense()
        B = np.array(B)
        C = np.array(C)
        U = B+C

    else:
        if ip=='5':
            W = (scipy.sparse.load_npz('./data/W_5.npz')).todense()
        elif ip=='3':
            W = (scipy.sparse.load_npz('./data/W_3.npz')).todense()
        else:
            W = (scipy.sparse.load_npz('./data/W_10.npz')).todense()
        W = np.array(W)
        U = W

    fname = 'IdxIdsMap.txt'
    idx2id = Helper.load(fname)

    cluster_centers = []
    cluster_center_vectors = []

    no_of_clusters = input("Please enter the number of clusters required -- ")
    no_of_clusters = int(no_of_clusters)

    for ii in range(no_of_clusters):
        center = np.random.randint(U.shape[0])
        cluster_centers.append(center)
        cluster_center_vectors.append(U[center])

    its=4

    clist = findclusters(U,cluster_center_vectors)

    while its!=1:
        cluster_center_vectors = updatecenters(U,clist)
        clist = findclusters(U,cluster_center_vectors)
        its-=1

    rlist = sort_closest_first(U,clist,cluster_center_vectors)

    for ii in range(len(rlist)):
    	for jj in range(len(rlist[ii])):
    		rlist[ii][jj] = idx2id[rlist[ii][jj]]


    cluster_dict = {}
    for ii in range(len(rlist)):
    	cluster_dict[str(ii)] = list(rlist[ii])

    config = cp.ConfigParser()
    config.read(os.getcwd()+'/config.ini')
    dpath = config['TASK2']['devset_path']

    a = True
    vis.showclusters(cluster_dict,dpath)

if __name__ == '__main__':
    main()
