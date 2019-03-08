from src import LocationMapping as lm
from src import ShowImages as si
import numpy as np
import math
import os
import csv
import configparser as cp

def vector_length(v):
	length = 0
	for element in v:
		prod = pow(element, 2)
		length = length + prod
	length = pow(length, 1/2)

	return length

def compute_hash(x, r):
	w = 4
	b = (-w) * np.random.random_sample() + w
	h = np.squeeze(np.floor( (np.dot(x.T,r) + b) / w))
	h = str(int(h))
	return h


def create_image_dictionary(config, models, location_map):
	
	imageFeatureMatrix = {}
	visual_descriptor_path = config['TASK5']['visual_descriptor_path']

	for k, v in location_map.items():
		for model in models: 
			file_name = visual_descriptor_path+"{} {}.csv".format(v, model)
			reader = csv.reader(open(file_name, 'r'))
			for row in reader:
				if row[0] in imageFeatureMatrix:  
					imageFeatureMatrix[row[0]].extend([float(i) for i in row[1:]])
				else:
					imageFeatureMatrix[row[0]] = [float(i) for i in row[1:]]

	return imageFeatureMatrix

def index(L, k, r, image_dictionary):

	index_tables = {}

	for image_id, image_vector in image_dictionary.items():

		if len(image_vector) > 263:
			continue

		for i in range(L):
			if i not in r:
				r[i] = {}
			if i not in index_tables:
				index_tables[i] = {}
			h = ''
			for j in range(k):
				if j not in r[i]:
					r_temp = np.random.randn(len(image_vector), 1)
					r_temp = r_temp / np.linalg.norm(r_temp)
					r[i][j] = r_temp
				h = h + compute_hash(np.array(image_vector).reshape(len(image_vector),1), r[i][j])
			if h not in index_tables[i]:
				index_tables[i][h] = []
			index_tables[i][h].append(image_id)

	return index_tables, r

def search(L, k, index_tables, r, query_image_id, image_dictionary):

	# print("query image", query_image_id)

	query_image_dictionary = {}
	query_image_dictionary[query_image_id] = image_dictionary[query_image_id]
	query_image_index, r = index(L, k, r, query_image_dictionary)

	# print("index", query_image_index)

	collision_images = []
	for i in range(L):
		query_image_hash = list([key for key in query_image_index[i]])[0]
		if query_image_hash not in index_tables[i]:
			continue
		collision_images.extend(index_tables[i][query_image_hash])

	return collision_images

def calculate_distance(imageFeatureMatrix, queryImage, documents, t):

	topTImages = []
	distances= []
	topTDistances = []
	images= []
	imagesDistances = {}
	flag = 0

	for image, features in imageFeatureMatrix.items():

		if image not in documents:
			continue

		queryVector = imageFeatureMatrix[queryImage]  
		distance = math.sqrt(sum([(float(a) - float(b)) ** 2 for a, b in zip(queryVector, features)])) 
		if len(distances) < int(t): 
			distances.append(distance)
			images.append(image)
			imagesDistances[image] = distance 
		else: 
			if flag == 0: 
				topTDistances = sorted(distances) 
				topTImages = [i[0] for i in sorted(imagesDistances.items(), key=lambda x: x[1])]
				flag = 1
			pointer = int(t) 
			for index in range(int(t)):
				if topTDistances[index] > distance:
					pointer = index 
					break
			if pointer < int(t):
				topTDistances = topTDistances[:pointer] + [distance] + topTDistances[pointer: int(t)-1] 
				topTImages = topTImages[:pointer] + [image] + topTImages[pointer: int(t)-1]

	return topTImages, topTDistances

def main():

	config = cp.ConfigParser()
	config.read(os.getcwd()+'/config.ini')

	devset_directory = config['TASK5']['devset_path']
	valid_models = ['CM','CN','CN3x3','LBP3x3']
	# valid_models = ['CM','CN','HOG','LBP3x3']

	print('creating location maps')
	location_dictionary = lm.mapLocationIdWithName(devset_directory)
	location_image_dictionary = lm.mapLocationNameWithImageId(devset_directory)

	print('creating image dictionary')
	image_dictionary = create_image_dictionary(config, valid_models, location_dictionary)

	L = int(input("enter L:\n"))
	k = int(input("enter k:\n"))

	print("creating index tables")
	index_tables, r = index(L, k, {}, image_dictionary)
	print("index tables created")

	image_id = input("enter image id:\n")

	print("searching")
	collision_images = search(L, k, index_tables, r, image_id, image_dictionary)
	print("total images collided with:", len(collision_images))
	print("total unique images:", len(set(collision_images)))
	collision_images = list(set(collision_images))
	collision_images.append(image_id)
	# collision_images = list([key for key in image_dictionary])

	t = int(input("enter t:\n"))

	print("calculating distance")
	images, distances = calculate_distance(image_dictionary, image_id, collision_images, t)
	
	print("visualising")
	si.fetchImages(images, devset_directory)

	# '5187687918'
	# '10045378096'
	# '753366817'

main()

