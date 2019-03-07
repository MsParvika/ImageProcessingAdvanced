from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

class Math:

	def svd(self, data, k):
		s = TruncatedSVD(n_components=k, n_iter=7, random_state=42)
		d = s.fit(data)
		components = d.components_
		ev = d.explained_variance_ratio_ 
		data = s.transform(data)
		return data, components, ev

	def pca(self, data, k):
		p = PCA(n_components=k)
		d = p.fit(data)
		components = d.components_
		ev = d.explained_variance_ratio_ 
		data = p.transform(data)
		return data, components, ev

	def lda(self, data, k):
		l = LatentDirichletAllocation(n_components=k, random_state=0, learning_method='batch', learning_decay=0.0)
		d = l.fit(data)
		components = d.components_
		ev = np.zeros((k,))
		data = l.transform(data)
		return data, components, ev

	def dot_product(self, v1, v2):

		element_count = 0
		dot_product = 0
		
		while element_count < len(v1):
			prod = v1[element_count] * v2[element_count]
			dot_product = dot_product + prod
			element_count = element_count + 1
		
		return dot_product
	
	def length(self, v):

		length = 0
		for element in v:
			prod = pow(element, 2)
			length = length + prod
		length = pow(length, 1/2)

		return length

	def euclidean_distance(self, v1, v2):
		
		element_count = 0
		distance = 0
		while element_count < len(v1):
			difference = pow(v1[element_count] - v2[element_count], 2)
			distance = distance + difference
			element_count = element_count + 1

		distance = pow(distance, 1/2)

		return distance


	def manhattan_distance(self, v1, v2):

		distance = 0
		for i in range(0, len(v1)):
			distance = distance + abs(v1[i]-v2[i])
		return distance

	def cosine_similarity(self, v1, v2):

		numerator = self.dot_product(v1, v2)
		length_v1 = self.length(v1)
		length_v2 = self.length(v2)
		denominator = length_v1 * length_v2
		similarity = numerator / denominator

		return similarity