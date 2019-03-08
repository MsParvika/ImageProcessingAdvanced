from PIL import Image
import LocationMapping as LM
import matplotlib.pyplot as plt
import numpy as np

def fetchImages(imageIds, dirPath):
    location_map = LM.mapLocationNameWithImageId(dirPath)
    images = []
    titles = []
    for id in imageIds:
        images.append(Image.open(dirPath + '/img/' + location_map[id] + '/' + id + '.jpg'))
        titles.append(location_map[id])
    cols = 4
    n_images = len(images)
    fig = plt.figure()
    for n, (image, locationName, id) in enumerate(zip(images, titles, imageIds)):
        print(n)
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        plt.imshow(image)
        a.set_title(locationName, fontsize=5)
        a.set_xlabel(id, fontsize=5)
        a.set_yticklabels([])
        a.set_xticklabels([])
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
