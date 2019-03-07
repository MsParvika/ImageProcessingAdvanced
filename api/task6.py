from api import helpers
from lib import math
import scipy
import numpy as np


class Task6:

	def __init__(self, init_params):
		self.config = init_params['config']
		self.project_directory = init_params['project_directory']
		self.math = math.Math()
		self.location_data = (scipy.sparse.load_npz(self.project_directory + self.config.get("PREPROCESSED-DATA", "location_features"))).todense() 

	def similarity_matrix(self):
		similarity_matrix = np.dot(self.location_data, self.location_data.T)
		return similarity_matrix

	def reduce(self, similarity_matrix, k):
		similarity_matrix, components, ev = self.math.svd(similarity_matrix, k)
		components = np.asarray(components)
		ev = np.asarray(ev)
		np.savetxt(self.project_directory+"/task6_comp.csv", components[:,0:k], delimiter=",")
		np.savetxt(self.project_directory+"/task6_ev.csv", ev[0:k], delimiter=",")
		return similarity_matrix