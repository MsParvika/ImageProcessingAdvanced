# Image Processing Advanced
The project experiments with
• text and image features,
• vector models, and
• similarity/distance measures
##### Not meant to be Copied

This is the main readme file. Please follow everything written in this and individual readme files (for each packages) for smooth merging of code.

**Getting Started**
Open a terminal. Browse into Project directory - Image Processing.
Copy Data set into the current directory.
Install redis - https://redis.io/topics/quickstart
Run setup.py file
To run the project:
·   	For Windows: Type "Main.py"
·   	For Mac and Linux: Type "python ./Main.py"
The code is divided into 3 sections - api, config and lib
Api - contains all the API for tasks 1 to 7
Config - all the hard-coded file paths/names are written in this
Lib - all the packages reside in this directory

**Project Structure**
Math.py file in lib folder:
This python file contains the mathematical operation which we are going to use throughout the entire phase.
It contains separate functions for SVD, PCA, and LDA which take a matrix and the value of k (number of latent semantics) as its parameters and returns the diagonal matrix of eigenvalues. For SVD, we have used the TruncatedSVD since it works well for sparse matrix and does not center the data before computation.
We have the functions for finding out the similarity between two vectors in the form of Euclidean distance and Cosine similarity where we are passing the 2 vectors as parameters.
Lastly, we also have functions for scalar product / inner product for 2 vectors and for finding out the norm/ length of the vector.


** Coding standards and Package Structure **

We will be using Python3 with Object Oriented Programming. Each file will have its own class suitable member variables and functions. For different files under packages, please refer to the README files of all packages

I briefly describe the function of each file/folder

	1. main.py - this will be the runner script that will call APIs
	2. setup.py - perform startup tasks. no need to worry about this yet, leave it empty
	3. api - contains all task classes and a helper class. refer to package readme for details
	4. config - contains app-wide config as ini file and a parser class to query it
	5. data - directory to story any raw files(like devset). should be in .gitignore
	6. lib - contains all libraries for db, file, cache, math. refer to package readme for details

If you are familiar with OOP in Python3(like I am), you can refer to the following sample code. It looks easy to pick up.

	class Dog:

	    # Class Attribute
	    species = 'mammal'

	    # Initializer / Instance Attributes
	    def __init__(self, name, age):
	        self.name = name
	        self.age = age

	    # instance method
	    def description(self):
	        return "{} is {} years old".format(self.name, self.age)

	    # instance method
	    def speak(self, sound):
	        return "{} says {}".format(self.name, sound)

		# Instantiate the Dog object
		mikey = Dog("Mikey", 6)

		# call our instance methods
		print(mikey.description())
		print(mikey.speak("Gruff Gruff"))

**Prerequisites**
Python 3 or better should be installed and configured on the system
RAM should be at least 8GB


**Input Format**
Inputs except Directory Path should be space separated.
Visual descriptors should be in upper case and rest all should be in lower case.
Command: Enter task Number
Input:1
Command: Enter value for  k
Input: 5  (space separated)

 
**Libraries Required**
numpy
xml.etree.ElementTree
sklearn.LDA
sklearn.PCA
sklearn.TruncatedSVD
pandas
