import os
import numpy as np
import scipy.io as scio
import sys
from typing import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sklearn as sk


def get_list_of_files(dirName: str, absolute=True) -> List[str]:
	"""
	This function creates a list of all the paths of .mat files inside a directory.list_of_file.
	
	:param dirName: a string representing a path
	:param absolute: a boolean, whether we want absolute paths or not
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
			all_files = all_files + get_list_of_files(full_path)
		else:
			if full_path.endswith(".mat"):
				if absolute:
					all_files.append(full_path)
				else:
					all_files.append(entry)
	return all_files


def load_list_of_files(listOfFiles: [str]) -> tuple({List[float], List[str]}):
	"""
	This function takes in argument a list of paths of .mat files and load them into a list.
	
	:param listOfFiles: a list of strings (the paths)
	:return: database: 2D array (a lot x 6)
	databaseNames: 1D string array of names
	"""

	database = []
	databaseNames = []
	# Iterate over all the entries
	for entry in listOfFiles:
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
	file_intel = file_name_analysis(filePath)
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
	"""
	This function creates a list of all numpy arrays loaded from a given list of paths.

	:param list_of_files: the list of all paths to load
	:return: all_files: the list of all loaded numpy arrays
	"""
	nb_of_files = len(list_of_files)

	big_data_matrix = []

	for i in range(nb_of_files):
		data_matrix_tmp = load_single_data(list_of_files[i])[0]
		for j in range(len(data_matrix_tmp)):
			big_data_matrix.append(data_matrix_tmp[j])

	return big_data_matrix


def file_name_analysis(filePath: str) -> dict:
	"""
	This function takes in argument path of a .mat file and load it into a list.
	
	:param filePath: a string, the path
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
	"""
	This function plots in 2D and 3D the values of 1 sensor of 1 recording in the dataframe "all_data_matrix"

	:param all_data_matrix: The list of all recordings (861 x ~160 x 9)
	:param trial_number: The trial coordinate of the desired recording
	:param subject_number: The subject coordinate of the desired recording
	:param action_number: The action coordinate of the desired recording
	:param sensor_id: The id of the desired sensor (1 for accelerometer, 2 for gyroscope)
	"""
	sub_dataframe = extract_recordings(all_data_matrix, action_number, subject_number, trial_number)
	# sub_dataframe = all_data_matrix
	trajectory3d = extract_sensor_data(sub_dataframe, sensor_id)
	plottable_trajectory = np.array(trajectory3d).T
	print("Trajectory to be plotted, size : {}".format(plottable_trajectory.shape))
	print("Just to be sure, the size of a line : {}".format(plottable_trajectory[0].shape))
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot(plottable_trajectory[0], plottable_trajectory[1], plottable_trajectory[2], marker='x')
	# ax.scatter(*plottable_trajectory.T[0], color='red')

	if sensor_id == 1:
		title = "Accelerometer values, time dependent"
		ylabel = "Accelerometer values, unknown unit"
	else:
		title = "Gyroscope values, time dependent"
		ylabel = "Gyroscope values, unknown unit"
	plt.figure(title)
	abscissa = [i / 50 for i in range(len(plottable_trajectory[0]))]
	plt.plot(abscissa, plottable_trajectory[0])
	plt.plot(abscissa, plottable_trajectory[1])
	plt.plot(abscissa, plottable_trajectory[2])
	plt.xlabel("time, in seconds")
	plt.ylabel(ylabel)
	plt.title(title)
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

	:param trial_number:
	:param subject_number:
	:param action_number:
	:param all_data_matrix: a matrix of size Nx9, containing the data to extract
	:return: sub_dataframe: the data extracted
	"""
	sub_dataframe = []
	for entry in all_data_matrix:
		if entry[-1] == action_number and entry[-2] == trial_number and entry[-3] == subject_number:
			sub_dataframe.append(entry)
	return sub_dataframe


def extract_sensor_data(dataframe: List, sensor_id: int) -> List[List[int]]:
	"""
	This function takes in argument a dataframe and the id of a sensor,a dn extract the trajectory of this sensor.
	sensor_id = 1 : accelerometer
	sensor_id = 2 : gyroscope
	sensor_id = 0 : both

	:param dataframe: a matrix of size Nx9, containing the data to extract
	:param sensor_id: 1 or 2, depending on the sensor desired
	:return: sub_dataframe: the data extracted
	"""
	sub_dataframe = []
	for i in range(len(dataframe)):
		extracted_tmp = [0] * 3
		if sensor_id == 0:
			extracted_tmp = dataframe[i][:6]
		elif sensor_id == 1:
			extracted_tmp = dataframe[i][:3]
		elif sensor_id == 2:
			extracted_tmp = dataframe[i][3:6]
		else:
			print("Incorrect sensor id given. Given {}, should be 1 or 2".format(sensor_id))
		sub_dataframe.append(extracted_tmp)
	return sub_dataframe


def compute_trajectory(sensor_data: List, origin: list) -> List[list]:
	"""
	This function computes an approximate trajectory from 1 recording.
	Currently working badly, only taking into account the acceleration.

	:param origin: The coordinates in 3D for the 2 first step
	:param sensor_data: The list of linear/angular accelerations
	"""
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
	for i in range(nb_points - 2):
		position_matrix.append(
			sensor_data[i] + position_matrix[i - 1] + position_matrix[i - 1] - position_matrix[i - 2])

	# print("Computed trajectory : ")
	# print(position_matrix)
	plottable_trajectory = []

	for i in range(3):
		plottable_column = []
		for j in range(nb_points):
			plottable_column.append(position_matrix[j][i])
		plottable_trajectory.append(plottable_column)

	print("Shape of the plotable matrix, should be 3xN : {}".format(np.array(plottable_trajectory).shape))

	fig = plt.figure('Real trajectory')
	ax = fig.gca(projection='3d')
	ax.plot(plottable_trajectory[0], plottable_trajectory[1], plottable_trajectory[2], marker='x')
	plt.show()

	return plottable_trajectory


def extract_single_features(dataframe: List) -> List[List[np.ndarray]]:
	"""
	Gets the mean and variance of each sensor data along each axis, for a single recording.
	Meaning : don't feed a whole database to this function, just 1 extracted recording.

	:param dataframe: a matrix of size ~160x9, the data
	:return: features: the extracted features (6 x 2)
	"""
	np_data_tr = np.array(dataframe).T
	features = []
	for i in range(6):
		features_tmp = [np.mean(np_data_tr[i]), np.std(np_data_tr[i])]
		features.append(features_tmp)
	return features


def extract_all_features(dataframe: List) -> Tuple[List[List[List[np.ndarray]]], Any]:
	"""
	Gets the mean and variance of each sensor data along each axis, for ALL recordings.

	:param dataframe: a matrix of size 861 x ~160 x 9, the data
	:return: all_features: the extracted features (861 x 6 x 2)
	:return: recording_lengths: the length of each recording (861 x 1)
	"""
	all_coordinates = []
	all_features = []
	recording_lengths = []
	for entry in dataframe:
		current_coordinates = entry[6:9]
		if current_coordinates not in all_coordinates:
			recording_lengths.append(1)
			all_coordinates.append(current_coordinates)
		else:
			recording_lengths[-1] = recording_lengths[-1] + 1
	print(all_coordinates)
	print(recording_lengths)
	for coordinate in all_coordinates:
		sub_dataframe = extract_recordings(dataframe, coordinate[2], coordinate[0], coordinate[1])
		feature_tmp = extract_single_features(sub_dataframe)
		all_features.append(feature_tmp)

	return all_features, recording_lengths


def split_dataframe(dataframe: List, train_id: List[int], normalize=True) -> Tuple[list, list, list, list]:
	"""
	THIS VERSION IS DEPRECATED and returns an unusable result.
	Use split_features instead.

	Splits a dataframe into a test set and a training set, according to the ids given.

	:param normalize: Whether the data should be normalized (mean = 0, std = 1)
	:param train_id: The ids (subject numbers) we want inside the training set
	:param dataframe: a matrix of size 861 x ~160 x 9, the data
	:return: train_set, test_set: The 2 dataframes of data, normalized (or not), with only the valuable data remaining
	:return: train_label, train_label: The 2 sets of labels
	"""
	nb_points = len(dataframe)
	all_features, recording_lengths = extract_all_features(dataframe)
	# dataframe = extract_sensor_data(dataframe, sensor_id=0)
	nb_recordings = len(recording_lengths)
	absolute_coordinate = 0
	if normalize:
		for recording in range(nb_recordings):
			for data in range(recording_lengths[recording]):
				for attribute in range(6):
					# print(absolute_coordinate, nb_points)
					# print(all_features[recording][0], all_features[recording][1])
					dataframe[absolute_coordinate][attribute] = (dataframe[absolute_coordinate][attribute] -
					                                             all_features[recording][0]) / all_features[recording][1]
				absolute_coordinate += 1
	test_set = []
	test_label = []
	train_set = []
	train_label = []
	absolute_coordinate = 0
	print("Shape of the dataframe before splitting : {}".format(np.array(dataframe).shape))
	print("Computed number of lines by recording : {}".format(recording_lengths))
	for i in range(nb_recordings):
		print("Splitting the dataframe {},{}".format(i, absolute_coordinate))
		if dataframe[absolute_coordinate][8] in train_id:
			print(np.array(dataframe[absolute_coordinate:absolute_coordinate+recording_lengths[i]]).shape)
			print("---------------------------------------------")
			train_set.append(dataframe[absolute_coordinate:absolute_coordinate+recording_lengths[i]])
			train_label.append(dataframe[absolute_coordinate][8])
			absolute_coordinate = absolute_coordinate + recording_lengths[i]
		else:
			test_set.append(dataframe[absolute_coordinate:absolute_coordinate + recording_lengths[i]])
			test_label.append(dataframe[absolute_coordinate][8])
			absolute_coordinate = absolute_coordinate + recording_lengths[i]

	for i in range(nb_recordings):
		print("Splitting the dataframe {},{}".format(i, absolute_coordinate))
		train_tmp = []
		test_tmp = []
		train_label_tmp = []
		test_label_tmp =[]
		for j in range(recording_lengths[i]):
			if dataframe[absolute_coordinate][8] in train_id:
				train_tmp.append(dataframe[absolute_coordinate][:6])
				train_label_tmp.append(dataframe[absolute_coordinate][8])
				absolute_coordinate = absolute_coordinate + 1
			else:
				test_tmp.append(dataframe[absolute_coordinate:absolute_coordinate + recording_lengths[i]])
				test_label_tmp.append(dataframe[absolute_coordinate][8])
				absolute_coordinate = absolute_coordinate + 1

	return train_set, train_label, test_set, test_label

def split_features(all_features: list, train_ids: List[int], normalize=True) -> Tuple[list,list,list,list]:
	good_to_go_features = []
	for i in range(len(all_features)):
		for j in range(len(all_features[i])/2):
			good_to_go_features.append(all_features[i][2*j])
			good_to_go_features.append(all_features[i][2*j +1])
	

def train_classifier(training_set: list, training_label: list, classifier):
	return classifier.fit(training_set, training_label)

if __name__ == "__main__":
	end = input("Program terminated without any error. Press any key to leave.")
