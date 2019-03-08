import numpy as np
import networkx as nx
from src import get_img_data_for_2 as gi
import configparser as cp
import Visualize as vis
import Helper as hp
import LocationMapping as LM
import os


image_ids,imgs,cos_img,sorted_k = gi.get_data()

k = input("Enter the number of nearest neighbours ")
k = int(k)

fname = 'IdxIdsMap.txt'
idx2id = hp.load(fname)

G = nx.Graph()
for ii in range(sorted_k.shape[0]): 
    for jj in range(k+1):
        # G.add_weighted_edges_from([(ii,sorted_k[ii][jj],cos_img[ii][int(sorted_k[ii][jj])])])
        G.add_edge(ii,sorted_k[ii][jj])

c = input("Enter the number of clusters required ")
c = int(c)

no_of_steps_away = []
cluster_reps = []

nn = np.random.randint(sorted_k.shape[0],size=100)

mx_degree = 0
for nnn in nn:
        degreee = G.degree(nnn)
        if degreee>mx_degree:
                indx_final = nnn
                mx_degree = degreee

cluster_reps.append(indx_final)

#Finding cluster reps
while c!=1:
        mndist = cos_img.shape[0] + 1 #Assuming a node can be at mox (no of total nodes) steps away from another node
        indx = []
        for ii in range(cos_img.shape[0]):
                dst = 0
                #Finding new cluster rep farthest to the existing ones that are stored in cluster_reps  
                for cluster_repsl in cluster_reps:
                        dst+=cos_img[cluster_repsl][ii]
                if mndist>dst:
                        indx = []
                        mndist = dst
                        indx.append(ii)
                elif mndist==dst:
                        indx.append(ii)
                else:
                        continue
        no_of_steps_away.append(mndist)
        mx_degree = 0
        indx_final = 0
        for inddx in indx:
                degreee = G.degree(inddx)
                if degreee>mx_degree:
                        indx_final = inddx
                        mx_degree = degreee
        cluster_reps.append(indx_final)
        c-=1

path_exists_for = [False]*cos_img.shape[0]


config = cp.ConfigParser()
config.read(os.getcwd()+'/config.ini')
dpath = config['TASK2']['devset_path']
img2loc = LM.mapLocationNameWithImageId(dpath)

rep_node_dict = {}
for ncluster_reps in cluster_reps:
        rep_node_dict[str(ncluster_reps)]=[]

min_node_distance_list = []
closest_cluster_list = []


for ii in range(cos_img.shape[0]):
        flag = False
        min_node_dist = cos_img.shape[0]
        temp = np.random.randint(len(cluster_reps),size=1)
        closest_cluster = cluster_reps[temp[0]]
        for cluster_repsl in cluster_reps:
                try:
                        tl = nx.shortest_path_length(G,source=ii,target=cluster_repsl)
                except nx.NetworkXNoPath:
                        tl = 0 
                if tl>0:
                        flag = flag or True
                if (tl<min_node_dist) and (tl!=0):
                        min_node_dist = tl
                        closest_cluster = cluster_repsl
                if flag == True:
                        path_exists_for[ii]=True
        min_node_distance_list.append(min_node_dist)
        closest_cluster_list.append(closest_cluster)
        rep_node_dict[str(closest_cluster)].append(ii)

sorted_dist_dict = {}  

for ncluster_reps in cluster_reps:
        sorted_dist_dict[ncluster_reps]=[]

#Assigning cluster reps to themselves
for ncluster_reps in cluster_reps:
        rep_node_dict[str(closest_cluster_list[ncluster_reps])].remove(ncluster_reps)
        min_node_distance_list[ncluster_reps]=0
        closest_cluster_list[ncluster_reps]=ncluster_reps
        rep_node_dict[str(ncluster_reps)].append(ncluster_reps)

#Sorting closest images to cluster representatives
for keys in sorted_dist_dict:
        for jj in range(len(rep_node_dict[str(keys)])):
                sorted_dist_dict[keys].append(min_node_distance_list[rep_node_dict[str(keys)][jj]])
        sorted_dist_dict[keys]=list(np.argsort(sorted_dist_dict[keys]))
        temp = [0]*len(rep_node_dict[str(keys)])
        for jj in range(len(rep_node_dict[str(keys)])):
                temp[jj]=rep_node_dict[str(keys)][sorted_dist_dict[keys][jj]]
        rep_node_dict[str(keys)]=temp


print("Cluster representatives are as follows: "+str(cluster_reps))

for key,value in rep_node_dict.items():
        for jj in range(len(rep_node_dict[key])):
            rep_node_dict[key][jj] = idx2id[rep_node_dict[key][jj]]
            
vis.showclusters(rep_node_dict,dpath, True)

