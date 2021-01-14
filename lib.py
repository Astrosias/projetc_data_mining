import os
import numpy as np
import scipy.io as scio
import sys
from typing import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


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
	# print(data)
	file_intel = fileNameAnalysis(filePath)
	nb_rows = len(data)
	nb_cols = len(data[0])
	print("Size of the datachunk : {}, size of each element : {}".format(nb_rows, nb_cols))
	print(file_intel)
	# full_data_matrix = np.zeros(shape = (nb_rows, nb_cols + 3))
	# full_data_matrix = [[0]*(nb_cols+3)]*nb_rows
	full_data_matrix = []
	for i in range(nb_rows):
		data_tmp = []
		for j in range(nb_cols):
			data_tmp.append(data[i, j])
			# full_data_matrix[i][j]= data[i,j]
			# print(i, j)

		# print("Debug inside load_single_data, values : {}".format(data_tmp))
		# full_data_matrix[i][nb_cols] = file_intel['subject_number']
		# full_data_matrix[i][nb_cols+1] = file_intel['trial_number']
		# full_data_matrix[i][nb_cols+2] = file_intel['action_number']
		data_tmp.append(file_intel['subject_number'])
		data_tmp.append(file_intel['trial_number'])
		data_tmp.append(file_intel['action_number'])
		full_data_matrix.append(data_tmp)
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
	:return: data: a dictionary with all the info on the file (subject, number, sensor)
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


def trace_signal(all_data_matrix: List, sensor_id: int, action_number: int, subject_number: int, trial_number: int):
	# sub_dataframe = extract_recordings(all_data_matrix, action_number, subject_number, trial_number)
	sub_dataframe = all_data_matrix
	trajectory3d = extract_sensor_data(sub_dataframe, sensor_id)
	plotable_trajectory = np.array(trajectory3d).T
	print("Trajectory to be plotted, size : {}".format(plotable_trajectory.shape))
	print("Just to be sure, the size of a line : {}".format(plotable_trajectory[0].shape))
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot(plotable_trajectory[0], plotable_trajectory[1], plotable_trajectory[2], marker='x')
	# ax.scatter(*plotable_trajectory.T[0], color='red')

	plt.figure('test')
	plt.plot(plotable_trajectory[0])
	plt.plot(plotable_trajectory[1])
	plt.plot(plotable_trajectory[2])

	plt.show()
	# fig = plt.figure()
	# ax = fig.gca(projection='3d')
	# theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
	# z = np.linspace(-2, 2, 100)
	# r = z ** 2 + 1
	# x = r * np.sin(theta)
	# y = r * np.cos(theta)
	# ax.plot(plotable_trajectory[0][:100], y, z, label='parametric curve')
	# ax.legend()
	#
	# plt.show()
	pass


def extract_recordings(all_data_matrix: List, action_number: int, subject_number: int, trial_number: int) -> List[
	float]:
	"""
	This function takes in argument a dataframe and the coordinates of a recording, and returns the corresponding lines.

	:param all_data_matrix: a matrix of size Nx9, containing the data to extract
	:return: sub_dataframe: the data extracted
	"""
	sub_dataframe = []
	for entry in all_data_matrix:
		if entry[-1] == action_number and entry[-2] == trial_number and entry[-3] == subject_number:
			sub_dataframe.append(entry)
	return sub_dataframe


def extract_sensor_data(dataframe: List, sensor_id: int) -> List[float]:
	"""
	This function takes in argument a dataframe and the id of a sensor,a dn extract the trajectory of this sensor.
	sensor_id = 1 : accelerometer
	sensor_id = 2 : gyroscope

	:param dataframe: a matrix of size Nx9, containing the data to extract
	:param sensor_id: 1 or 2, depending on the sensor desired
	:return: sub_dataframe: the data extracted
	"""
	sub_dataframe = []
	for i in range(len(dataframe)):
		extracted_tmp = [0] * 3
		for j in range(3):
			if sensor_id == 1:
				extracted_tmp[j] = dataframe[i][j]
			elif sensor_id == 2:
				extracted_tmp[j] = dataframe[i][j + 3]
			else:
				print("Incorrect sensor id given. Given {}, should be 1 or 2".format(sensor_id))
		sub_dataframe.append(extracted_tmp)
	return sub_dataframe


def compute_trajectory(sensor_data: List, origin: list) -> List[list]:

	nb_points = len(sensor_data)
	# acceleration_matrix = []
	# for i in range(3):
	# 	acceleration_matrix.append([sensor_data[i][0] for i in range(nb_points)])
	# position_matrix = np.zeros(shape=(nb_points, 3))
	for i in range(2):
		for j in range(2):
			sensor_data[i][j] = origin[i][j]
	sensor_data = np.array(sensor_data)

	position_matrix = []
	for i in range(2):
		position_matrix.append(origin[i])
	for i in range(nb_points-2):
		position_matrix.append(sensor_data[i] + position_matrix[i-1] + position_matrix[i-1] - position_matrix[i-2])


	print("Computed trajectory : ")
	print(position_matrix)
	plotable_trajectory = []

	for i in range(3):
		plotable_column = []
		for j in range(nb_points):
			plotable_column.append(position_matrix[j][i])
		plotable_trajectory.append(plotable_column)

	print("Shape of the plotable matrix, should be 3xN : {}".format(np.array(plotable_trajectory).shape))

	fig = plt.figure('Real trajectory')
	ax = fig.gca(projection='3d')
	ax.plot(plotable_trajectory[0], plotable_trajectory[1], plotable_trajectory[2], marker='x')
	plt.show()

	return plotable_trajectory


if __name__ == "__main__":
	end = input("Program terminated without any error. Press any key to leave.")
