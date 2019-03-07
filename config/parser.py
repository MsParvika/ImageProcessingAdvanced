import configparser as cp

class ConfigParser:
	def __init__(self, path):
		self.config = cp.ConfigParser()
		self.config.read(path)

	# return value given key of given section
	def get(self, section, key):
		return self.config[section][key]
