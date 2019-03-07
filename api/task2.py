from lib import cache
from lib import math
from api import helpers
from xml.dom import minidom
import numpy as np
from operator import itemgetter

class Task2:

	def __init__(self, init_params, data):
		
		self.config = init_params['config']
		self.app_mode = init_params['app_mode']
		self.project_directory = init_params['project_directory']
		self.cache = cache.Cache( self.config.get("CACHE", "host"), int(self.config.get("CACHE", "port")) )
		self.math = math.Math()
		self.maps = self.generate_maps()
		self.data = data


	def similarity(self, query, limit, vector_space):

		self_similar_vectors = self.self_similarity(query, limit, vector_space)
		other_similar_vectors = self.other_similarity(query, limit, vector_space)

		return self_similar_vectors, other_similar_vectors


	def self_similarity(self, query_id, limit, vector_space):

		ids = self.cache.hgetall("dm_"+vector_space)
		indices = self.cache.hgetall("idm_"+vector_space)

		if vector_space == "location":
			query_id = self.cache.hgetall("inverse_location_map")[query_id.encode('utf-8')].decode('utf-8')
		print(query_id)

		query = self.data[vector_space][int(ids[query_id.encode('utf-8')].decode('utf-8'))]

		similar_vectors = []

		for i in range(len(self.data[vector_space])):
			similarity = {}
			similarity['value'] = self.math.cosine_similarity(self.data[vector_space][i], query)
			similarity['index'] = str(i)
			similarity['id'] = indices[str(i).encode('utf-8')]
			similar_vectors = helpers.sort(similar_vectors, similarity, limit, 'value', 0)

		return similar_vectors


	def other_similarity(self, query_id, limit, vector_space):

		other_vector_spaces = ['user', 'location', 'image']
		other_vector_spaces.remove(vector_space)

		queries = {}
		for space in other_vector_spaces:
			queries[space] = self.maps[vector_space][space]
		
		query_matrices = []

		for key, value in queries.items():
			query_map = {}
			query_map["space"] = key
			query_map["data"] = self.get_v1(key, value[query_id], self.cache.hgetall("dm_"+key))
			query_matrices.append(query_map)

		sim = []
		for query_matrix in query_matrices:
			sim.append(self.find_closest(query_matrix['space'], query_matrix['data'], self.data[query_matrix['space']], limit))

		return sim

	
	def get_v1(self, key, ids, m):

		indices = []
		for i in ids:
			if key == "location":
				i = self.cache.hgetall("inverse_location_map")[i.encode('utf-8')].decode('utf-8')
			indices.append(int(m[i.encode('utf-8')].decode('utf-8')))
		
		data = []
		for i in indices:
			row = np.ravel(self.data[key][i])
			data.append(row)

		return data



	def generate_maps(self):

		maps = {}
		maps['user'] = {}
		maps['image'] = {}
		maps['location'] = {}
		
		image_to_user = {}
		user_to_image = {}
		image_to_location = {}
		location_to_image = {}
		location_to_user = {}
		user_to_location = {}
		
		xml_directory = helpers.load_directory(self.project_directory + self.config.get("RAW-DATA", "xmls"))		

		for file in xml_directory:
		   
		    filename = helpers.get_file_name(file)
		    location_id = self.cache.hgetall('location_map')[filename.encode('utf-8')].decode('utf-8')

		    mydoc = minidom.parse(file)
		    photos = mydoc.getElementsByTagName('photo')
		    
		    for elem in photos:
		    	image_to_user[elem.attributes['id'].value] = []
		    	image_to_location[elem.attributes['id'].value] = []
		    	user_to_location[elem.attributes['userid'].value] = []
		    	user_to_image[elem.attributes['userid'].value] = []
		    	location_to_image[location_id] = []
		    	location_to_user[location_id] = []

		    for elem in photos:
		    	image_to_user[elem.attributes['id'].value].append(elem.attributes['userid'].value) 
		    	image_to_location[elem.attributes['id'].value].append(location_id)
		    	if location_id not in user_to_location[elem.attributes['userid'].value]:
		    		user_to_location[elem.attributes['userid'].value].append(location_id)
		    	user_to_image[elem.attributes['userid'].value].append(elem.attributes['id'].value)
		    	location_to_image[location_id].append(elem.attributes['id'].value )
		    	if elem.attributes['userid'].value not in location_to_user[location_id]:
		    		location_to_user[location_id].append(elem.attributes['userid'].value )

		maps['user']['location'] = user_to_location
		maps['user']['image'] = user_to_image
		maps['image']['user'] = image_to_user
		maps['image']['location'] = image_to_location
		maps['location']['user'] = location_to_user
		maps['location']['image'] = location_to_image

		return maps


	def find_closest(self, key, v1, v2, k):
		
		minDist = []

		average = np.zeros((len(v1[0],)))
		for i in range(0, len(v1)):
			average = average + v1[i]
		average = average / len(v1)

		for j in range(0, len(v2)):
			val = self.math.cosine_similarity(average, np.ravel(v2[j]))
			minDist.append({"distance": val, "term1":i, "term2":j})
			minDist = helpers.sort(minDist, {"distance": val, "term1":i, "term2":j}, k, 'distance', 0)
		count = 0
		l = []
		while (count < k):
			if minDist[count]["term2"] not in l:
				temp = {}
				temp['distance'] = minDist[count]['distance']
				temp['compare_id'] = self.cache.hgetall("idm_"+key)[str(minDist[count]['term2']).encode('utf-8')]
				l.append(temp)
				count = count + 1
		return l













