from lib import cache
from lib import math
from api import helpers
import scipy.sparse
import numpy as np
import time

class Task3:

	def __init__(self, init_params, data, image_location_map):

		self.config = init_params['config']
		self.app_mode = init_params['app_mode']
		self.project_directory = init_params['project_directory']
		self.cache = cache.Cache( self.config.get("CACHE", "host"), int(self.config.get("CACHE", "port")) )
		self.math = math.Math() 
		self.data = data
		self.image_location_map = image_location_map


	def reduce(self, model, algo, k):

		self.model = model

		if algo == "lda":
			self.data = self.reduce_lda(model, k)

		reduction_function = {}
		reduction_function['svd'] = self.math.svd
		reduction_function['pca'] = self.math.pca
		reduction_function['lda'] = self.math.lda

		reduction_data = []
		image_count = {}
		for location_id in self.data:
			image_count[location_id] = len( self.data[location_id][self.model] )
			reduction_data.extend(self.data[location_id][self.model])

		reduction_data = np.squeeze(reduction_data)
		total_reduced_data, components, ev = reduction_function[algo](reduction_data, k)

		components = np.asarray(components)
		np.savetxt(self.project_directory+"/task3.csv", components[:,:k], delimiter=",")
		
		self.reduced_data = {}
		prev_index = 0
		for key, value in image_count.items():
			self.reduced_data[key] = total_reduced_data[prev_index:prev_index+value]
			prev_index = prev_index + value


	def reduce_lda(self, model, k):

		files = helpers.load_directory(self.project_directory + self.config.get("PREPROCESSED-DATA","visual_directory"))
		data = {}

		for filepath in files:

			filename = helpers.get_file_name(filepath)
			file_model = filename.split()[1]
			location_name = filename.split()[0]
			location_id = self.cache.hgetall('location_map')[location_name.encode('utf-8')].decode('utf-8')

			if model != file_model:
				continue

			file = (scipy.sparse.load_npz(filepath)).todense()

			if location_id not in data:
				data[location_id] = {}
			if file_model not in data[location_id]:
				data[location_id][file_model] = []

			data[location_id][file_model] = file

		return data



	def similarity(self, image_id, limit):

		self.image_id = image_id
		self.location_id = self.image_location_map[self.image_id][0]
		self.limit = limit

		image_similarity = self.calculate_image_similarity()
		location_similarity = self.location_similarity()

		return image_similarity, location_similarity


	def calculate_image_similarity(self):

		print(self.location_id)
		print(self.model)
		print(self.image_id)

		query_vector_index = int( self.cache.get(self.location_id+'_'+self.model+"_"+self.image_id) )
		query_vector = self.reduced_data[self.location_id][query_vector_index]

		similar_vectors = []

		for locations, images in self.reduced_data.items():

			image_index = 0
			for image in images:

				s = {}
				s['query_id'] = self.image_id
				s['compare_id'] = self.cache.get(self.location_id+'_'+self.model+"_"+str(image_index)) 
				s['value'] = self.math.manhattan_distance(query_vector, image)

				similar_vectors = helpers.sort(similar_vectors, s, self.limit, 'value', 1)

				image_index = image_index + 1

		return similar_vectors


	def location_similarity(self):

		query_matrix = self.reduced_data[self.location_id]
		similar_vectors = []

		for k, v in self.reduced_data.items():
			s = {}
			s['query_id'] = self.location_id
			s['compare_id'] = k
			s['value'] = self.min_pair_similarity(query_matrix, v)
			similar_vectors = helpers.sort(similar_vectors, s, self.limit, 'value', 1)

		return similar_vectors


	def min_pair_similarity(self, m1, m2):

		matrix_distance = 0.0

		for r1 in m1:
			minimum = float("inf")
			for r2 in m2:
				dist = self.math.manhattan_distance(r1, r2)	
				if dist < minimum:
					minimum = dist
			matrix_distance = matrix_distance + minimum

		return matrix_distance/len(m1)


