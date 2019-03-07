from api import helpers
from lib import cache
from lib import math 
import scipy
import time
import numpy as np

class Task1:

	def __init__(self, init_params):

		self.config = init_params['config']
		self.app_mode = init_params['app_mode']
		self.project_directory = init_params['project_directory']
		self.cache = cache.Cache( self.config.get("CACHE", "host"), int(self.config.get("CACHE", "port")) )
		self.math = math.Math()
	
		self.user_data = (scipy.sparse.load_npz(self.project_directory + self.config.get("PREPROCESSED-DATA", "user_features"))).todense()
		self.image_data = (scipy.sparse.load_npz(self.project_directory + self.config.get("PREPROCESSED-DATA", "image_features"))).todense()
		self.location_data = (scipy.sparse.load_npz(self.project_directory + self.config.get("PREPROCESSED-DATA", "location_features"))).todense()

		self.visual_descriptors = self.generate_visual_descriptors_map()

	def reduce_dims(self, algo, k):

		textual_vector_spaces = ['user', 'image', 'location']
		textual_data = [self.user_data, self.image_data, self.location_data]

		reduction_function = {}
		reduction_function['svd'] = self.math.svd
		reduction_function['pca'] = self.math.pca
		reduction_function['lda'] = self.math.lda

		self.reduced_textual = {}

		for i in range(0, len(textual_vector_spaces)):

			if textual_vector_spaces[i] not in self.reduced_textual:
				self.reduced_textual[textual_vector_spaces[i]] = {}

			self.reduced_textual[textual_vector_spaces[i]], components, ev = reduction_function[algo](textual_data[i], k)
			components = np.asarray(components)
			ev = np.asarray(ev)
			np.savetxt(self.project_directory+"/task1_comp_"+str(i)+".csv", components[:,0:k], delimiter=",")
			np.savetxt(self.project_directory+"/task1_ev_"+str(i)+".csv", ev[0:k], delimiter=",")

	
	def generate_visual_descriptors_map(self):

		visual_descriptors = {}
		visual_descriptors_directory = helpers.load_directory(self.project_directory + self.config.get("RAW-DATA", "visual_descriptors"))

		for file in visual_descriptors_directory:
			
			location_name = helpers.get_file_name(file).split()[0]
			location_id = self.cache.hgetall('location_map')[location_name.encode('utf-8')].decode('utf-8')
			model = helpers.get_file_name(file).split()[1]
			
			if location_id not in visual_descriptors:
				visual_descriptors[location_id] = {}
			if model not in visual_descriptors[location_id]:
				visual_descriptors[location_id][model] = []

			file = helpers.load_text_file(file)
			lines = []
			for line in file:
				line = helpers.tokenize(line, ",")
				line = helpers.to_float(line[1:len(line)-1])
				lines.append(line)
			
			visual_descriptors[location_id][model] = lines

		return visual_descriptors





		



