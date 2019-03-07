App wide configuration elements are to be written in config.ini file. Please do NOT hardcode them. Anything ranging from db/collection name to application mode should be present here. A sample config file is provided.

This package contains two files:

	config.ini - has all configurable elements as key-value pairs
	parser.py - class that contains methods to load and get key values

A sample code is written below:

	from config import config_parser as cp

	config = cp("/path/to/config/file")
	db_port = cp.get("DB", "port")

