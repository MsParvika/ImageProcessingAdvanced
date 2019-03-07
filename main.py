import api.task1 as task1
import api.task2 as task2
import api.task3 as task3
import api.task4 as task4
import api.task5 as task5
import api.task6 as task6
import api.task7 as task7
from config import parser
import numpy as np
import scipy.sparse
import os

project_directory = os.getcwd()
config_file_path = project_directory + '/config/config.ini'

init_params = {}
init_params['config'] = parser.ConfigParser(config_file_path)
init_params['app_mode'] = init_params['config'].get("APP", "mode")
init_params['project_directory'] = os.getcwd()

t1 = task1.Task1(init_params)
maps = {}

choice = "yes"

while choice == "yes":

	task_choice = input("which task would you like to run?\n")

	if task_choice == "1":
		
		vector_space = input("enter vector space:\n")
		reduction_algo = input("enter reduction algorithm:\n")
		k = input("enter k:\n")

		t1 = task1.Task1(init_params)
		t1.reduce_dims(reduction_algo, int(k))

		print(t1.reduced_textual[vector_space])
		print('\n')

		t2_choice = input("do you want to continue with task2 (yes/no)?\n")
		if t2_choice == "no":
			continue

		vector_space_id = input("enter vector space id:\n")
		t2 = task2.Task2(init_params, t1.reduced_textual)
		maps = t2.maps
		s, o = t2.similarity(vector_space_id, 5, vector_space)

		print("similarity with itself:\n")
		print(s)
		print('\n')
		print("similarity with other vector spaces:\n")
		print(o)
		print('\n')


	elif task_choice == "3":

		model = input("enter model:\n")
		reduction_algo = input("enter reduction algorithm:\n")
		k = input("enter k:\n")
		
		t3 = task3.Task3(init_params, t1.visual_descriptors, maps['image']['location'])
		t3.reduce(model, reduction_algo, int(k))

		image_id = input("enter image id:\n")
		s_images, s_locations = t3.similarity(image_id, 30)
		print(s_images)
		print(s_locations)
		print('\n')

	elif task_choice == "4":

		model = input("enter model:\n")
		reduction_algo = input("enter reduction algorithm:\n")
		k = input("enter k:\n")

		t4 = task4.Task4(init_params, t1.visual_descriptors)
		t4.reduce(model, reduction_algo, int(k))

		location_id = input("enter location id:\n")
		s = t4.similarity(location_id, 5)
		print(s)
		print('\n')

	elif task_choice == '5':

		t5 = task5.Task5(init_params, t1.visual_descriptors)

		reduction_algo = input("enter reduction algorithm:\n")
		k = input("enter k:\n")
		location_id = input("enter location id:\n")

		ranks = t5.similarity(["CM", "CM3x3", "CN", "HOG", "LBP", "LBP3x3", "GLRLM", "GLRLM3x3", "CSD", "CN3x3"], reduction_algo, int(k), location_id, 5)
		print(ranks)
		print('\n')

	elif task_choice == '6':

		k = input("enter k:\n")

		t6 = task6.Task6(init_params)
		similarity_matrix = t6.similarity_matrix()
		reduced_similarity_matrix = t6.reduce(similarity_matrix, int(k))

		print(reduced_similarity_matrix)
		print('\n')

	elif task_choice == '7':
		k = input("enter k:\n")
		t7 = task7.Task7(init_params, int(k))

	choice = input("would you like to continue (yes/no)\n")
