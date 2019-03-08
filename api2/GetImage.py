from os import path
import numpy as np
import json
import time
import configparser as cp
import os

config = cp.ConfigParser();
config.read(os.getcwd()+'/config.ini')
filePath = config['TASK2']['image_term_path']

def get_data():
        start = time.time()
        if path.exists("imgvectors.txt")!=True:
                #Open file
                f = open(filePath,'r', encoding="utf8")
                o = f.readline()
                p=[]
                #Read every line of the file
                while o!="":
                        p.append(o)
                        o=f.readline()
                terms = []
                imglist = []

                #Scanned every stored line, made a list of the unique terms in every line
                for ip in p:
                        sp = ip.split(" ")
                        for k in range(1,len(sp)-1,4):
                                if sp[k] not in terms:
                                        terms.append(sp[k])

                imgs = []
                image_ids=[]
                #Scanned every stored line
                for ip in p:
                        idf = [0]*len(terms)
                        idx = []
                        sp=ip.split(" ")
                        #Stored the index of the encountered keyword in 'sp' from the 'terms' list, and used the index value to store the corresponding idf of the keyword in the respective list
                        for k in range(1,len(sp)-1,4):
                                idx.append(terms.index(sp[k]))
                                idf[idx[len(idx)-1]] = float(sp[k+3])
                        imgs.append(idf)
                        image_ids.append(sp[0])
                f.close()


                f1 = open('imgvectors.txt','w')
                f2 = open('imgids.txt','w')
                json.dump(imgs,f1)
                json.dump(image_ids,f2)
                f1.close()
                f2.close()
                image_ids=np.asarray(image_ids)
                imgs=np.asarray(imgs)
                print ('Image vector and ids file saved')

        else:
                f1 = open('imgvectors.txt','r')
                f2 = open('imgids.txt','r')

                image_ids = json.load(f2)
                imgs = json.load(f1)
                image_ids=np.asarray(image_ids)
                imgs=np.asarray(imgs)

                f1.close()
                f2.close()

        print ("Time elapsed in reading/creating file -- "+str(time.time() - start)+" seconds")

        print("Images vector shape-- "+str(imgs.shape))

        if path.exists("img_cosine_matrix.csv")!=True:
                cov_img = np.dot(imgs,imgs.T)
                print(cov_img.shape)

                cos_img = np.full([cov_img.shape[0],cov_img.shape[1]], np.nan)

                for ii in range(cov_img.shape[0]):
                        for jj in range(ii,cov_img.shape[1]):
                                cos_img[ii][jj] = cov_img[ii][jj]/np.sqrt(cov_img[ii][ii]*cov_img[jj][jj])
                        if ii%1000 == 0:
                                print (ii)
                                print (cos_img[ii][ii])
                                end = time.time()
                                print ("Time elapsed -- " + str(end - start)+" seconds")

                for ii in range(cos_img.shape[0]):
                        for jj in range(0,ii):
                                cos_img[ii][jj]=cos_img[jj][ii]

                end = time.time()
                print ("Time elapsed after copying upper triangle elements -- " + str(end - start)+" seconds")

                print("Searching NaN (should be []): "+str(np.argwhere(np.isnan(cos_img))))
                np.savetxt("img_cosine_matrix.csv", cos_img, delimiter=",", fmt='%1.8f')
                print("Cosine matrix file saved")


        else:
                cos_img = np.genfromtxt('img_cosine_matrix.csv', delimiter=',',dtype='float')
                print ("Time elapsed "+str(time.time()-start))


        if path.exists("sorted_indices_upto_100.csv")!=True:
                k=100
                sorted_k = np.full([cov_img.shape[0],k+1], np.nan)

                for ii in range(sorted_k.shape[0]):
                        sk = np.flip(np.argsort(cos_img[ii]))
                        for jj in range(0,k+1):
                                sorted_k[ii][jj]=int(sk[jj])

                print('Sorted indices for the first-indexed image '+str(sorted_k[0]))

                np.savetxt("sorted_indices_upto_100.csv", sorted_k, delimiter=",", fmt='% 4d')
                print('Sorted indices exported -- total time elapsed'+str(time.time()-start))


        else:
                sorted_k = np.genfromtxt('sorted_indices_upto_100.csv', delimiter=',', dtype='int')
                print('Sorted indices for the first-indexed image '+str(sorted_k[0]))

                print ("Time elapsed "+str(time.time()-start))

        return image_ids,imgs,cos_img,sorted_k
