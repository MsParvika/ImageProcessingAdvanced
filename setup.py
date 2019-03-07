from config import parser
from api import helpers
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import redis
import glob
import cv2
import sys
import os
import math
import time

def form_collections(project_directory, config):

	textual_descriptors = {}
	textual_descriptors['user'] = helpers.load_text_file(project_directory + config.get("RAW-DATA", "user_features"))
	textual_descriptors['image'] = helpers.load_text_file(project_directory + config.get("RAW-DATA", "image_features"))
	textual_descriptors['location'] = helpers.load_text_file(project_directory + config.get("RAW-DATA", "location_features"))

	location_keys_file = helpers.load_xml_file(project_directory + config.get("RAW-DATA", "location_keys"))
	location_keys_map = create_location_keys_map(location_keys_file)

	textual_collections = {}
	for key, file in textual_descriptors.items():
		print("generating for:", key)
		textual_collections[key] = generate_textual_descriptors(project_directory, config, key, file, location_keys_map)

	return textual_collections, location_keys_map


def create_location_keys_map(location_keys_file):

	location_keys_map = {}
	topics = location_keys_file.getroot()

	for topic in topics:
		location_keys_map[topic[1].text] = topic[0].text

	return location_keys_map


def generate_textual_descriptors(project_directory, config, key, input_file, location_keys_map):

	feature_map = {}
	feature_index = -1
	data_map = {}
	data_index = -1
	inverse_data_map = {}

	for line in input_file:
		
		tokens = helpers.tokenize(line, " ")

		data_index = data_index + 1
		data_map[tokens[0]] = data_index
		inverse_data_map[data_index] = tokens[0]

		i = get_starting_index(key, tokens, location_keys_map)
		while i < len(tokens):
			if tokens[i] not in feature_map:
				feature_index = feature_index + 1
				feature_map[tokens[i]] = feature_index
			i = i + 4

	data_feature_matrix = np.zeros( ( len(data_map.items()), len(feature_map.items()) ) )
	data_index = -1

	for line in input_file:

		data_index = data_index + 1
		feature_vector = np.zeros( (len(feature_map.items())) )

		tokens = helpers.tokenize(line, " ")

		i = get_starting_index(key, tokens, location_keys_map)
		while i < len(tokens)-1:
			feature = tokens[i]
			feature_index = feature_map[feature]
			feature_vector[feature_index] = float(tokens[i+3])
			i = i + 4

		data_feature_matrix[data_index] = feature_vector

	sparse_data_matrix = scipy.sparse.csr_matrix(data_feature_matrix)
	scipy.sparse.save_npz(project_directory + config.get('PREPROCESSED-DATA','textual_directory')+key+'.npz', sparse_data_matrix)

	r = redis.Redis(host=config.get('CACHE','host'), port=int(config.get('CACHE','port')))

	r.hmset('dm_'+key, data_map)
	r.hmset('idm_'+key, inverse_data_map)
	r.hmset('location_map', location_keys_map)
	r.hmset('inverse_location_map', reverse_map(location_keys_map))
		

def reverse_map(map):
	new_map = {}
	for key, value in map.items():
		new_map[value] = key
	return new_map


def get_starting_index(key, tokens, location_keys_map):

	start = 1
	if key == "location":
			location_name = tokens[0]
			location_name_length = len(helpers.tokenize(location_name, "_"))
			if location_name == 'doge_s_palace':
				location_name_length = 2
			start = start + location_name_length

	return start


def normalize_visual_descriptors(project_directory, config, location_keys_map):

	raw_file_directory = project_directory + config.get('RAW-DATA','visual_descriptors')
	preprocessed_file_directory = project_directory + config.get('PREPROCESSED-DATA','visual_directory')

	files = helpers.load_directory(raw_file_directory)

	for filepath in files:

		temp_visual_descriptor = []
		file = helpers.load_text_file(filepath)
		filename = helpers.get_file_name(filepath)
		model = filename.split()[1]

		print("generating for:",filename)

		if "CM" in model:
			cm(preprocessed_file_directory, filename, file)
			continue
		elif "CN" in model:
			cn(preprocessed_file_directory, filename, file)
			continue
		elif "HOG" in model:
			hog(preprocessed_file_directory, filename, file)
			continue
		elif "LBP3x3" in model:
			lbp(preprocessed_file_directory, filename, file, 1000)
		elif "LBP" in model:
			lbp(preprocessed_file_directory, filename, file, 100)
			continue
		elif "CSD" in model:
			csd(preprocessed_file_directory, filename, file)
			continue
		elif "GLRLM" in model:
			glrlm(preprocessed_file_directory, filename, file)
			continue

def cm(preprocessed_file_directory, filename, file):

	temp_visual_descriptor = []
	for line in file:
		line = helpers.tokenize(line, ",")
		line = line[1:4]
		line = helpers.to_float(line)
		line = np.array(line)
		line.astype(np.uint8)
		temp_visual_descriptor.append(line)

	visual_descriptor = np.asarray(temp_visual_descriptor)
	np.savetxt(preprocessed_file_directory+filename+".csv", visual_descriptor, delimiter=",")

	sparse_data_matrix = scipy.sparse.csr_matrix(temp_visual_descriptor)
	scipy.sparse.save_npz(preprocessed_file_directory+filename+'.npz', sparse_data_matrix)

def cn(preprocessed_file_directory, filename, file):

	temp_visual_descriptor = []
	for line in file:
		line = helpers.tokenize(line, ",")
		line = line[1:len(line)]
		line = np.array(np.array(helpers.to_float(line)) * 100)
		line.astype(np.uint8)
		temp_visual_descriptor.append(line)

	sparse_data_matrix = scipy.sparse.csr_matrix(temp_visual_descriptor)
	scipy.sparse.save_npz(preprocessed_file_directory+filename+'.npz', sparse_data_matrix)

def hog(preprocessed_file_directory, filename, file):

	temp_visual_descriptor = []
	for line in file:

		line = helpers.tokenize(line, ",")
		line = line[1:len(line)]
		line = helpers.to_float(line)
		line = np.array(line)

		for i in range(0, len(line), 9):
			line[i:i+9] = line[i:i+9] / sum(line[i:i+9])

		line = line * 100

		line.astype(np.uint8)
		temp_visual_descriptor.append(line)

	sparse_data_matrix = scipy.sparse.csr_matrix(temp_visual_descriptor)
	scipy.sparse.save_npz(preprocessed_file_directory+filename+'.npz', sparse_data_matrix)

def lbp(preprocessed_file_directory, filename, file, multiplying_factor):

	temp_visual_descriptor = []
	for line in file:
		line = helpers.tokenize(line, ",")
		line = line[1:len(line)]
		line = np.array(np.array(helpers.to_float(line)) * multiplying_factor)
		line = np.array(line)
		line.astype(np.uint8)
		temp_visual_descriptor.append(line)

	sparse_data_matrix = scipy.sparse.csr_matrix(temp_visual_descriptor)
	scipy.sparse.save_npz(preprocessed_file_directory+filename+'.npz', sparse_data_matrix)

def csd(preprocessed_file_directory, filename, file):

	temp_visual_descriptor = []
	for line in file:
		line = helpers.tokenize(line, ",")
		line = line[1:len(line)]
		line = helpers.to_float(line)
		line = np.reshape(cv2.normalize(np.array(line), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F), (len(line)))
		line = np.floor(line)
		line.astype(np.uint8)
		temp_visual_descriptor.append(line)

	sparse_data_matrix = scipy.sparse.csr_matrix(temp_visual_descriptor)
	scipy.sparse.save_npz(preprocessed_file_directory+filename+'.npz', sparse_data_matrix)

def glrlm(preprocessed_file_directory, filename, file):

	temp_visual_descriptor = []
	for line in file:
		line = helpers.tokenize(line, ",")
		line = line[1:len(line)]
		line = helpers.to_float(line)
		line = np.array(line)
		line.astype(np.uint8)
		temp_visual_descriptor.append(line)

	temp_visual_descriptor = np.array(temp_visual_descriptor)

	for i in range(0, len(temp_visual_descriptor)):
		for j in range(0, len(temp_visual_descriptor[0])):
			temp_visual_descriptor[i][j] = np.floor( ( temp_visual_descriptor[i][j] / max(temp_visual_descriptor[:,j]) ) * 100 ) 
		

	sparse_data_matrix = scipy.sparse.csr_matrix(temp_visual_descriptor)
	scipy.sparse.save_npz(preprocessed_file_directory+filename+'.npz', sparse_data_matrix)


def create_visual_keys(project_directory, config, location_keys_map):

	raw_file_directory = project_directory + config.get('RAW-DATA','visual_descriptors')
	preprocessed_file_directory = project_directory + config.get('PREPROCESSED-DATA','visual_directory')
	files = helpers.load_directory(raw_file_directory)

	r = redis.Redis(host=config.get('CACHE','host'), port=int(config.get('CACHE','port')))

	for filepath in files:

		location_name = helpers.tokenize(helpers.get_file_name(filepath), " ")[0]
		model = helpers.tokenize(helpers.get_file_name(filepath), " ")[1]
		location_id = location_keys_map[location_name]

		file = helpers.load_text_file(filepath)

		index = 0
		for line in file:
			
			image_id = helpers.tokenize(line, ",")[0]
			
			key = location_id + "_" + model + "_" + image_id
			value = str(index)
			r.set(key, value)

			key = location_id + "_" + model + "_" + str(index)
			value = str(image_id)
			r.set(key, value)

			index = index + 1


def main():

	localtime = time.asctime( time.localtime(time.time()) )
	print(localtime)

	print("initialising project")

	project_directory = os.getcwd()
	config_file_path = project_directory + '/config/config.ini'

	print("reading config")
	config = parser.ConfigParser(config_file_path)

	print("generating textual descriptors feature vectors")
	textual_collections, location_keys_map = form_collections(project_directory, config)

	print("generating visual descriptors feature vectors")
	visual_descriptors = normalize_visual_descriptors(project_directory, config, location_keys_map)

	print("initialising cache")
	create_visual_keys(project_directory, config, location_keys_map)

	localtime = time.asctime( time.localtime(time.time()) )
	print(localtime)
	print("done at:", localtime)
	print("thank you for your patience")

main()
