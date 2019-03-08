from operator import itemgetter
from src import Helper as hp
from src import SpareMatrixPR as SMPR
from collections import OrderedDict;
from src import visualize  as vis
import configparser as cp
import os

config = cp.ConfigParser()
config.read(os.getcwd()+'/config.ini')
image_textual_descriptor_fileName = config['TASK6']['raw_image_path'];

def make_dictionary():
    file_object=open(image_textual_descriptor_fileName, encoding="utf8");

    #reading the entire txt file line by line
    content= file_object.readlines()
    print (len(content))

    images = {}
    for x in content:

        # one image with words as a list of strings, numbers
        temp_image = x
        name = temp_image[0:temp_image.index('"')].strip()
        temp_image = temp_image[temp_image.index('"'):].replace('"',"").split()
        if name not in images.keys():
            images[name] = {}
        for i in range(3,len(temp_image),4):
            t = temp_image[i-3]
            images[name][t] = float(temp_image[i])
    return images

def read_input(img_dict):

    #sample input file path
    file_object = open("input_task_6.txt", "r")

    # reading the entire txt file line by line
    content = file_object.readlines()
    images = {}
    labels = []
    for x in content:
        if len(x.split()) != 0:
            line = x.split()
            name = line[0]
            if name in img_dict.keys():
                temp_image = line[1:]
                images[name] = {}
                images[name]["dict"] = img_dict[name]
                for y in temp_image:
                    images[name]["label"] = y
                    if y not in labels:
                        labels.append(y)
    return images,labels

def cosine_similarity_task_6(imgA,imgB):
    dot_prod = 0
    Sa = 0
    Sb = 0
    if len(imgA.keys())<len(imgB.keys()):
        temp1 = imgA
        temp2 = imgB
    else:
        temp1 = imgB
        temp2 = imgA
    for key1, value1 in temp2.items():
        Sb += value1 ** 2

    for key,value in temp1.items():
        Sa += value ** 2
        if key in temp2.keys():
            dot_prod += temp1[key] * temp2[key]
    return dot_prod/((Sa**0.5)*(Sb**0.5))


adj_matrix = hp.load("adj_matrix.txt")

# imageId to Index
IdsIdxMap = hp.load("IdsIdxMap.txt");

# Index to Image Id
IdxIdsMap = hp.load("IdxIdsMap.txt");


img_dict = make_dictionary()
img_label,labels = read_input(img_dict)
one_category_ids = []


imageidxlists = {}

for l in labels:
    imageidxlists[l] = []

for i in img_label.keys():
    l = img_label[i]["label"]
    row = IdsIdxMap[i]
    imageidxlists[l].append(row)

final_img_label = {}
knn_label_wise_images = {}


inp = int(input("Enter 1 for knn:"))

if inp == 1:
    for l in labels:
        knn_label_wise_images[l] = []

    k = int(input("Enter k: "))

    for key1 in img_dict.keys():
        sim = []
        final_img_label[key1] = {}
        final_img_label[key1] = {}

        for l in labels:
            final_img_label[key1][l] = 0

        for key2 in img_label.keys():
            x={}
            x["similarity"] = cosine_similarity_task_6(img_dict[key1],img_label[key2]["dict"])
            x["label"] = img_label[key2]["label"]
            sim.append(x)
            #if x["similarity"]>.999:
                #print(key1,key2,x["label"])
        sim = sorted(sim, key=itemgetter('similarity'),reverse=True)
        sim = sim [0:k]

        for x in sim:
            final_img_label[key1][x["label"]] += 1/k
            final_img_label[key1]["val"] = x["similarity"]
    final_img_label = OrderedDict(sorted(final_img_label.items(), key=lambda x: x[1]['val'], reverse=True))
    for keys in final_img_label.keys():
        del final_img_label[keys]["val"]
    for f,v in final_img_label.items():
        x = max(v.items(), key=itemgetter(1))[0]
        knn_label_wise_images[x].append(f)


    #give the img file path here
    devset_path = config['TASK6']['devset_path'];
    vis.showclusters(knn_label_wise_images, devset_path)


else:
    pprs = {}
    i = 0
    final_img_label_ppr = {}
    for x in img_dict.keys():
        final_img_label_ppr[x] = {}
        final_img_label_ppr[x]["val"] = -1

    for key in imageidxlists.keys():
        print (key)
        pprs[key] = SMPR.pagerank_scipy(adj_matrix, max_iter=1000, alpha=0.85, tol=1.0e-10, personalised=True, indxs=imageidxlists[key])

    ppr_label_wise_images = {}
    final_clusters = {}
    for label in pprs.keys():
        ppr_label_wise_images[label] = []
        final_clusters[label] = []
        for x in range(len(pprs[label])):
            y=pprs[label][x]
            if y>final_img_label_ppr[IdxIdsMap[x]]["val"]:
                final_img_label_ppr[IdxIdsMap[x]]["val"] = y
                final_img_label_ppr[IdxIdsMap[x]]["label"] = label

    sorted_final_img_label_ppr = sorted(final_img_label_ppr.items(), key=lambda x: x[1]['val'], reverse=True)


    for x in sorted_final_img_label_ppr:
        ppr_label_wise_images[x[1]["label"]].append(x[0])

    devset_path = config['TASK6']['devset_path'];
    vis.showclusters(ppr_label_wise_images,devset_path);