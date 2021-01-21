import sys
current_path = r'/home/alexandre/Documents/AADA/projetc_data_mining'
if current_path not in sys.path:
	sys.path.insert(0, r'/home/alexandre/Documents/AADA/projetc_data_mining')
import lib_data_mining
import numpy as np
import scipy.io as scio
import os
from sklearn import tree

filePath = "database/test/a1_s1_t1_inertial.mat"
dirName = "database/test/"

list_all_files = lib_data_mining.get_list_of_files(dirName)

big_dataset = lib_data_mining.load_all_data(list_all_files)

# print(big_dataset)

# lib_data_mining.trace_signal(big_dataset, 1, 1, 1, 1)
#
# origin = [[0, 0, 0], [0, 0, 0]]

# sensor_data = list(lib_data_mining.extract_sensor_data(big_dataset, 1))
# lib_data_mining.compute_trajectory(sensor_data, origin)

# test, test1 = lib_data_mining.extract_all_features(big_dataset)

train_set, train_label, test_set, test_label = lib_data_mining.split_dataframe(big_dataset, [1,27], normalize=True)
print("Size of the training set : {}".format(np.array(train_set).shape))
print("Size of the training labels : {}".format(len(train_label)))
# data = [[0, 0], [1, 1]]
# labels = [0, 1]
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(data, labels)
#
# print(clf.predict([[0,0.5]]))

classifier = tree.DecisionTreeClassifier()
#
# classifier = lib_data_mining.train_classifier(train_set, train_label, classifier)

classifier.fit(train_set, train_label)


if __name__ == "__main__":
	end = input("Program terminated without any error. Press any key to leave.")
