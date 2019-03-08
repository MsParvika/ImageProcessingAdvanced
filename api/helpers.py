import glob
import xml.etree.ElementTree as ET
import pickle;



def save(variable, flieName):
    with open(flieName, 'wb') as f:
        pickle.dump(variable, f);


def load(flieName):
    with open(flieName, 'rb') as f:
        b = pickle.load(f)
        return b;

def sort(arr, new_object, max_length, sort_key, bit):

	if len(arr) <= 0:
		arr.append(new_object)
		return arr
	
	i = 0
	while i < max_length and i < len(arr):
		condition = new_object[sort_key] > arr[i][sort_key]
		if bit == 1:
			condition = new_object[sort_key] < arr[i][sort_key]
		if condition:
			temp = arr.copy()
			temp[i] = new_object
			if max_length<=len(arr):
				temp[i+1 : max_length] = arr[i : max_length-1]
			else:
				temp[i+1 : len(temp)] = arr[i : len(arr)]
			arr = temp
			break
		i = i + 1

	if len(arr) < max_length and i == len(arr):
		arr.append(new_object)
		
	return arr


def load_directory(file_path):
	files=glob.glob(file_path)
	return files


def get_file_name(file_path):
	file_path = file_path.split("/")
	full_name = file_path[len(file_path)-1]
	name = full_name.split(".")[0]
	return name


def load_text_file(file_path):
	file = open(file_path, 'r')
	return file.readlines()


def load_xml_file(file_path):
	tree = ET.parse(file_path)
	return tree


def tokenize(text, separator):
	return text.split(separator)


def to_float(string_list):
	float_list = []
	for item in string_list:
		float_list.append(float(item))
	return float_list


def file_to_float(lines):
	file = []
	for line in lines:
		line = to_float(tokenize(line, ","))
		file.append(line)
	return file
