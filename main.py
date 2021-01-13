
import lib
import numpy as np
import scipy.io as scio
import sys
import os


filePath = "database/test/a1_s1_t1_inertial.mat"
dirName = "database/test/"

list_all_files = lib.getListOfFiles(dirName)

big_dataset = lib.load_all_data(list_all_files)

print(big_dataset)


if __name__ == "__main__":

	end = input("Program terminated without any error. Press any key to leave.")