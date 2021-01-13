import os
import numpy as np
import scipy.io as scio
import sys
from typing import *


def getListOfFiles(dirName: str, absolute=True) -> List[str]:
	"""
	This function creates a list of all the paths of .mat files inside a directory.list_of_file.
	
	:param dirName: a string representing a path
	:param absolute: a boolean, weither we want absolute paths or not
	:return: all_files: list of strings, the paths
	"""

	list_of_file = os.listdir(dirName)
	all_files = list()
	# Iterate over all the entries
	for entry in list_of_file:
	# Create full path
		full_path = os.path.join(dirName, entry)
	# If entry is a directory then get the list of files in this directory
		if os.path.isdir(full_path):
			all_files = all_files + getListOfWavFiles(full_path)
		else:
			if full_path.endswith(".mat"):
				if absolute:
					all_files.append(full_path)
				else:
					all_files.append(entry)
	return all_files


def loadListOfFiles(listOfFiles: [str]) -> tuple({List[float], List[str]}):
	"""
	This function takes in argument a list of paths of .mat files and load them into a list.
	
	:param listOfFiles: a list of strings (the paths)
	:return: database: 2D array (a lot x 6)
	databaseNames: 1D string array of names
	"""

	database = []
	databaseNames = []
	# Iterate over all the entries
	for entry in listOfFile:
		fileTmp = scio.loadmat(entry)
		database.append(fileTmp['d_iner'])
		databaseNames.append(os.path.basename(entry))
	return database, databaseNames


def load_single_data(filePath: str) -> tuple({List[float], dict}):
	"""
	This function takes in argument path of a .mat file and load it into a list.
	
	:param filePath: a string, the path
	:return: data: 2D array (a lot x 6), the data loaded
	"""
	data = scio.loadmat(filePath)['d_iner']
	file_intel = fileNameAnalysis(filePath)
	nb_rows = len(data)
	nb_cols = len(data[0])
	print("Size of the datachunk : {}, size of each element : {}".format(nb_rows, nb_cols))
	print(file_intel)
	# full_data_matrix = np.zeros(shape = (nb_rows, nb_cols + 3))
	full_data_matrix = [[0]*(nb_cols+3)]*nb_rows
	for i in range(nb_rows):
		for j in range(nb_cols):
			full_data_matrix[i][j]= data[i,j]
		full_data_matrix[i][nb_cols] = file_intel['subject_number']
		full_data_matrix[i][nb_cols+1] = file_intel['trial_number']
		full_data_matrix[i][nb_cols+2] = file_intel['action_number']
	return full_data_matrix, file_intel

def load_all_data(list_of_files: List[str]) -> List[np.array]:

	nb_of_files = len(list_of_files)

	big_data_matrix = []

	for i in range(nb_of_files):
		data_matrix_tmp = load_single_data(list_of_files[i])[0]
		for j in range(len(data_matrix_tmp)):
			big_data_matrix.append(data_matrix_tmp[j])

	return big_data_matrix

def fileNameAnalysis(filePath: str) -> dict:
	"""
	This function takes in argument path of a .mat file and load it into a list.
	
	:param filePAth: a lstring, the path
	:return: data: a dictionnary with all the info on the file (subject, number, sensor)
	"""
	allInfo = dict()
	fileName = os.path.basename(filePath)
	infoTmp = ''
	i = 1
	while fileName[i] != '_':
		infoTmp += fileName[i]
		i += 1
	allInfo['action_number'] = int(infoTmp)
	infoTmp = ''
	i += 2
	while fileName[i] != '_':
		infoTmp += fileName[i]
		i += 1
	allInfo['subject_number'] = int(infoTmp)
	infoTmp = ''
	i += 2
	while fileName[i] != '_':
		infoTmp += fileName[i]
		i += 1
	allInfo['trial_number'] = int(infoTmp)

	# print(allInfo)

	return allInfo

def trace_signal():
	pass

if __name__ == "__main__":
	end = input("Program terminated without any error. Press any key to leave.")
