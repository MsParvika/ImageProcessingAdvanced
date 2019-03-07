from api import helpers
import itertools
import time
import numpy as np
from tensorly.decomposition import parafac
from sklearn.cluster import KMeans

class Task7:

	def __init__(self, init_params, k):

		self.config = init_params['config']
		self.project_directory = init_params['project_directory']

		textual_descriptors = {}
		textual_descriptors['user'] = helpers.load_text_file(self.project_directory + self.config.get("RAW-DATA", "user_features"))
		textual_descriptors['image'] = helpers.load_text_file(self.project_directory + self.config.get("RAW-DATA", "image_features"))
		textual_descriptors['location'] = helpers.load_text_file(self.project_directory + self.config.get("RAW-DATA", "location_features"))

		location_keys_file = helpers.load_xml_file(self.project_directory + self.config.get("RAW-DATA", "location_keys"))
		self.location_keys_map = self.create_location_keys_map(location_keys_file)

		self.feature_list = {}
		self.lengths = {}
		for key, value in textual_descriptors.items():
			self.feature_list[key], self.lengths[key] = self.generate_feature_list(key, value)

		tuples = self.create_tuples()
		self.create_tensor(tuples, k)


	def create_location_keys_map(self, location_keys_file):

		location_keys_map = {}
		topics = location_keys_file.getroot()

		for topic in topics:
			location_keys_map[topic[1].text] = topic[0].text

		return location_keys_map


	def generate_feature_list(self, vector_space, file):

		feature_list = {}
		data_index = 0

		for line in file:
			
			tokens = helpers.tokenize(line, " ")
			i = self.get_starting_index(vector_space, tokens)

			while i < len(tokens)-1:
				feature = tokens[i]
				if feature not in feature_list:
					feature_list[feature] = []
				feature_list[feature].append(data_index)
				i = i + 4

			data_index = data_index + 1

		return feature_list, data_index


	def get_starting_index(self, vector_space, tokens):
		start = 1
		if vector_space == "location":
				location_name = tokens[0]
				location_name_length = len(helpers.tokenize(location_name, "_"))
				if location_name == 'doge_s_palace':
					location_name_length = 2
				start = start + location_name_length

		return start


	def create_tuples(self):

		tuples = []
		feature_count = 0
		for feature, indices in self.feature_list['user'].items():
			if (feature in self.feature_list['image']) and  (feature in self.feature_list['location']):
				print(feature, feature_count)
				feature_count = feature_count + 1
				tuples.extend(self.combine(self.feature_list['user'][feature], self.feature_list['image'][feature], self.feature_list['location'][feature]))

		return tuples


	def combine(self, v1, v2, v3):
		print(len(v1), len(v2), len(v3))
		a = [v1, v2, v3]
		return list(itertools.product(*a))


	def create_tensor(self, tuples, k):

		m, l, k = self.lengths['user'], self.lengths['image'], self.lengths['location']
		
		tensor = np.zeros((m,l,k))

		for i in range(0, len(tuples)-1):
			tensor[tuples[i][0]][tuples[i][1]][tuples[i][2]] += 1

		factors = parafac(tensor, rank=k, init='random', tol=10e-2)

		k = []
		for j in range(len(factors)):
			wcss = []
			slope = 1;
			slopeA = [1, 1, 1, 1, 1];
			i = 1
			val = 1;
			while True:
				kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
				kmeans.fit(factors[j])
				wcss.append(kmeans.inertia_)
				if i > 5:
					slope, intercept = np.polyfit(range(1, i + 1), wcss, 1);
					slopeA.append(slope);
					term1 = abs(slopeA[i - 1])
					term2 = abs(slopeA[i - 2])
					val = (term1 - term2) / term2;
					print(i);
				if len(factors[j]) <= i:
					break;
				if abs(val) < 0.05:
					break
				i += 1;
			k.append(i)

		# plt.plot(range(1, 10), wcss)
		# plt.title('the elbow method')
		# plt.xlabel('Number of clusters')
		# plt.show();
		print(k);
		# print(elapsed_time);
		final = []
		for x in range(len(k)):
			kmeans = KMeans(n_clusters=k[x], init='k-means++', max_iter=300, n_init=10, random_state=0)
			final.append(kmeans.fit_transform(factors[j]))
		# print(final[0][0], "\n\n\n", final[1][0])
		print(final)

