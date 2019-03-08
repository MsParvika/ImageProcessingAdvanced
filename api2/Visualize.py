from PIL import Image
import LocationMapping as LM
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

def showclusters(ppr_final_label_images, dirPath,lbs=False):
    location_map = LM.mapLocationNameWithImageId(dirPath)
    images = []
    titles = []
    maxl = -1
    for l,v in ppr_final_label_images.items():
        if len(v)>maxl:
            maxl = len(v)
    for id in range(maxl):
        fig  = plt.figure()
        x=1
        for label in ppr_final_label_images.keys():
            if id<len(ppr_final_label_images[label]):
                i = str(ppr_final_label_images[label][id])
                a = fig.add_subplot(1,len(ppr_final_label_images.keys()),x)
                plt.imshow(Image.open(dirPath + '/img/' + location_map[i] + '/' + i + '.jpg'))
                a.set_xlabel(i, fontsize=12)
                if lbs==False:
                    a.set_title(str(label),fontsize=12)
                else:
                    a.set_title(location_map[i],fontsize=12)
                x+=1
        fig.set_size_inches(np.array(fig.get_size_inches()) * len(ppr_final_label_images.keys()))
        plt.show()
