from api import task4

class Task5:

	def __init__(self, init_params, data):

		self.config = init_params['config']
		self.app_mode = init_params['app_mode']
		self.t4 = task4.Task4(init_params, data)


	def similarity(self, models, algo, k, location_id, limit):

		ranks = {}

		for model in models:

			self.t4.reduce(model, algo, k)
			similar_vectors = self.t4.similarity(location_id, 30)

			print(model, similar_vectors)

			for i in range(0, len(similar_vectors)-1):
				l_id = similar_vectors[i]['compare_id']
				if l_id not in ranks:
					ranks[l_id] = i
					continue
				ranks[l_id] = ranks[l_id] + i
			print(ranks)

		print(ranks)

		ranks = [(k, ranks[k]) for k in sorted(ranks, key=ranks.get, reverse=False)]

		return ranks

