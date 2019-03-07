import redis

class Cache:

	def __init__(self, host, port):
		self.conn = redis.Redis(host=host, port=port)

	def get(self, key):
		return self.conn.get(key)

	def sef(self, key, value):
		self.conn.set(key, value)

	def hgetall(self, key):
		return self.conn.hgetall(key)

	def hmset(self, key, value):
		self.conn.hmset