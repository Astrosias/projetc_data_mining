
import lib
import numpy as np
import scipy.io as scio
import sys
import os


filePath = "database/test/a1_s1_t1_inertial.mat"
dirName = "database/test/"

list_all_files = lib.getListOfFiles(dirName)

big_dataset = lib.load_all_data(list_all_files)

# print(big_dataset)

# lib.trace_signal(big_dataset, 1, 1, 1, 1)

origin = [[0, 0, 0], [0, 0, 0]]

sensor_data = list(lib.extract_sensor_data(big_dataset, 1))
lib.compute_trajectory(sensor_data, origin)


if __name__ == "__main__":

	end = input("Program terminated without any error. Press any key to leave.")
