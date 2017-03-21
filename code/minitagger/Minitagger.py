import os
import sys
import math
import time
import pickle
import random
import datetime
import subprocess
import collections
import numpy as np
from sklearn.model_selection import KFold
from name_lists_handler import build_name_lists

from SequenceData import SequenceData
from model_evaluation import report_fscore
from sklearn.metrics import confusion_matrix
from visualize_utils import plot_confusion_matrix
from utils import postprocess_predictions

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.constraints import maxnorm
from keras.layers import Dropout
from keras import backend
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.initializers import glorot_uniform


import sklearn_crfsuite

# for reproducibility
np.random.seed(6)

LIBLINEAR_PATH = os.path.join(os.path.dirname(__file__), "liblinear-1.96/python")
sys.path.append(os.path.abspath(LIBLINEAR_PATH))
import liblinearutil


class Minitagger(object):
	"""
	Represents the Minitagger model and can be used to train a classifier and make predictions.
	Also it includes the active learning feature
	"""

	def __init__(self):
		# feature extractor that is used (it is a SequenceDataFeatureExtractor object)
		self.__feature_extractor = None
		# it stores a trained liblinearutil
		self.__liblinear_model = None
		# CRF classifier
		self.__crf_classifier = None
		# NN classifier
		self.__nn_classifier = None
		# flag in order to print more/less log messages
		self.quiet = False
		# path to output directory for active learning
		self.active_output_path = ""
		# store predictions
		self.verbose = False
		# flag for cross validation
		self.cross_val = False
		# path to output the predictions
		self.prediction_path = ""
		# number of seed examples for active learning
		self.active_seed_size = 0
		# number of examples for labeling at each iteration in active learning
		self.active_step_size = 0
		# output actively selected examples every time this value divides their number
		self.active_output_interval = 0
		# classifier
		self.classifier = None
		# lists to keep exact, inexact and CONLL f-score for cross validation
		self.exact_fscore_list = []
		self.inexact_fscore_list = []
		self.conll_fscore_list = []

	def equip_feature_extractor(self, feature_extractor):
		"""
		Equips the Minitagger with a feature extractor

		@type feature_extractor: SequenceDataFeatureExtractor
		@param feature_extractor: contains the feature extraction object
		"""

		self.__feature_extractor = feature_extractor

	def __report_performance(self, data_test, pred_labels):
		"""
		Creates log files useful for debugging and prints a confusion matrix

		@type data_test: SequenceData object
		@param data_test: contains the testing data set
		@type pred_labels: list
		@param pred_labels: contains the prediction labels as they result from the classifier
		"""

		# file to print all predictions
		file_name = os.path.join(self.prediction_path, "predictions.txt")
		f1 = open(file_name, "w")
		# file to print only sentences that contain at least one wrong label after classification
		file_name = os.path.join(self.prediction_path, "predictions_wrong.txt")
		f2 = open(file_name, "w")
		# file to print only sentences whose labels are predicted 100% correctly
		file_name = os.path.join(self.prediction_path, "predictions_correct.txt")
		f3 = open(file_name, "w")
		sequence_number = -1
		for words, labels in data_test.sequence_pairs:
			sequence_number += 1
			for i in range(len(words)):
				# create line to print in the file
				line = words[i] + " " + labels[i] + " " + pred_labels[sequence_number][i] + "\n"
				# write to file
				f1.write(line)
			# separate sentences with empty lines
			f1.write("\n")
			if labels != pred_labels[sequence_number]:
				for i in range(len(words)):
					# create line to print in the file
					line = words[i] + " " + labels[i] + " " + pred_labels[sequence_number][i] + "\n"
					# write to file
					f2.write(line)
				# separate sentences with empty lines
				f2.write("\n")
			if labels == pred_labels[sequence_number]:
				for i in range(len(words)):
					# create line to print in the file
					line = words[i] + " " + labels[i] + " " + pred_labels[sequence_number][i] + "\n"
					# write to file
					f3.write(line)
				# separate sentences with empty lines
				f3.write("\n")
		f1.close()
		f2.close()
		f3.close()
		# print()
		# if not self.cross_val:
		# 	for label in ["B", "I", "O"]:
		# 		count = (np.array(true_labels) == label).sum()
		# 		print("Number of " + label + " in the test set:", count)
		# 	print()
		# 	# create confusion matrix
		# 	cm = confusion_matrix(true_labels, pred_labels)
		# 	print("--------------------- Confusion Matrix ---------------------\n")
		# 	for row in cm:
		# 		print(row)
		# 	print()
		file_name = os.path.join(self.prediction_path, "predictions.txt")
		modified_file_name = os.path.join(self.prediction_path, "modified_predictions.txt")
		postprocess_predictions(file_name, modified_file_name)
		exact_fscore, inexact_fscore, conll_fscore = report_fscore(modified_file_name, self.cross_val)
		# report f-scores
		print("Exact f-scrore: {0:.3f}".format(exact_fscore))
		print("Inexact f-scrore: {0:.3f}".format(inexact_fscore))
		print("CONLL f-scrore: {0:.3f}".format(conll_fscore))
		if self.cross_val:
			self.exact_fscore_list.append(exact_fscore)
			self.inexact_fscore_list.append(inexact_fscore)
			self.conll_fscore_list.append(conll_fscore)

	def __fit_and_predict_crf(self, data_train, data_test):
		"""
		Fits CRF model on the given train data and predict on given test data
		Reports performance if test data is given

		@type data_train: SequenceData
		@param data_train: the training data set
		@type data_test: SequenceData
		@param data_test: the test data set
		"""

		# Extract features only for labeled instances from data_train
		[label_list, features_list, _] = self.__feature_extractor.extract_features(data_train, False, [], self.classifier)
		# print some useful information about the data
		if (not self.quiet) and (not self.cross_val):
			print("{0} labeled words (out of {1})".format(len(label_list), data_train.num_of_words))
			print("{0} label types".format(len(data_train.label_count)))
			print("{0} word types".format(len(data_train.word_count)))
			print("\"{0}\" feature template".format(self.__feature_extractor.feature_template))
		# define problem to be trained using the parameters received from the feature_extractor
		self.__crf_classifier.fit(features_list, label_list)
		self.__feature_extractor.is_training = False
		[label_list, features_list, _] = self.__feature_extractor.extract_features(data_test, False, [], self.classifier)
		# make predictions using the CRF classifier
		pred_labels = self.__crf_classifier.predict(features_list)
		if self.verbose:
			# report performance achieved using the CRF classifier
			self.__report_performance(data_test, pred_labels)

	def __fit_and_predict_svm(self, data_train, data_test):
		"""
		Fits SVM model on the given train data and predict on given test data
		Reports performance if test data is given

		@type data_train: SequenceData
		@param data_train: the training data set
		@type data_test: SequenceData
		@param data_test: the test data set
		"""

		# keep the training start timestamp
		start_time = time.time()
		# reset the feature extractor dictionaries
		self.__feature_extractor.reset_feature_extractor()
		# Extract features only for labeled instances from data_train
		[label_list, features_list, _] = self.__feature_extractor.extract_features(data_train, False, [], self.classifier)
		# print some useful information about the data
		if (not self.quiet) and (not self.cross_val):
			print("{0} labeled words (out of {1})".format(len(label_list), data_train.num_of_words))
			print("{0} label types".format(len(data_train.label_count)))
			print("{0} word types".format(len(data_train.word_count)))
			print("\"{0}\" feature template".format(self.__feature_extractor.feature_template))
			print("{0} feature types".format(self.__feature_extractor.num_feature_types()))
		# define problem to be trained using the parameters received from the feature_extractor
		problem = liblinearutil.problem(label_list, features_list)
		# train the model (-q stands for quiet = True in the liblinearutil)
		self.__liblinear_model = liblinearutil.train(problem, liblinearutil.parameter("-q"))
		# training is done, set is_training to False, so that prediction can be done
		self.__feature_extractor.is_training = False
		# print some useful information
		if not self.quiet:
			num_seconds = int(math.ceil(time.time() - start_time))
			# how much did the training last
			print("Training time: {0}".format(str(datetime.timedelta(seconds=num_seconds))))
			# perform prediction on the data_test and report accuracy
			if (data_test is not None) or cv:
				quiet_value = self.quiet
				self.quiet = True
				pred_labels, acc = self.predict(data_test)
				self.quiet = quiet_value
				print("Test set accuracy: {0:.3f}%".format(acc))

	def __build_nn(self, num_features, hidden_units):
		"""
		Build NN model

		@type num_features: int
		@param num_features: number of features
		@type hidden_units: list
		@param hidden_units: number of neurons per hidden layer
		"""

		# define model
		self.__nn_classifier = Sequential()
		# initializer
		initializer = glorot_uniform(1234)
		# define input_dim and number of units in the next hidden layer (e.g. 32)
		self.__nn_classifier.add(Dense(hidden_units[0], input_dim=num_features, init=initializer, activation="relu", W_constraint=maxnorm(3)))
		if len(hidden_units) == 2:
			# add dropout
			self.__nn_classifier.add(Dropout(0.2, seed=1223))
			# add 2nd hidden layer
			self.__nn_classifier.add(Dense(hidden_units[1], init=initializer, activation="relu", W_constraint=maxnorm(3)))
		# define output layer with 3 classes and softmax activation function
		self.__nn_classifier.add(Dense(3, init=initializer, activation="softmax"))
		# Compile model
		adam = Adam(lr=0.001, decay=1e-3)
		self.__nn_classifier.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])

	def __fit_and_predict_nn(self, data_train, data_test):
		"""
		Fits NN model on the given train data and predict on given test data
		Reports performance if test data is given

		@type data_train: SequenceData
		@param data_train: the training data set
		@type data_test: SequenceData
		@param data_test: the test data set
		"""

		# keep the training start timestamp
		start_time = time.time()
		# reset the feature extractor dictionaries
		self.__feature_extractor.reset_feature_extractor()
		# Extract features only for labeled instances from data_train
		[label_list, features_list, _] = self.__feature_extractor.extract_features(data_train, False, [], self.classifier)
		# print some useful information about the data
		if (not self.quiet) and (not self.cross_val):
			print("{0} labeled words (out of {1})".format(len(label_list), data_train.num_of_words))
			print("{0} label types".format(len(data_train.label_count)))
			print("{0} word types".format(len(data_train.word_count)))
			print("\"{0}\" feature template".format(self.__feature_extractor.feature_template))
			print("{0} feature types".format(self.__feature_extractor.num_feature_types()))
		# get number of features
		num_features = self.__feature_extractor.num_feature_types()
		# get number of data points
		num_datapoints = len(features_list)
		# build X matrix for input to the NN: X = num_datapoints x num_features
		X_train = np.zeros([num_datapoints, num_features])
		for i, features_dict in enumerate(features_list):
			for key, value in features_dict.items():
				# set values in the X matrix according to the dictionary
				X_train[i, key - 1] = value
		# the labels [1,2,3] should be converted to [0,1,2]
		label_list = np.array(label_list) - 1
		# the labels should be converted to categorical values (0 becomes [0 0 0], 1 becomes [0 1 0], etc)
		y_train = to_categorical(label_list)

		# build model
		# self.__build_nn(num_features, [128, 64])
		# # define early stopping parameter
		# early_stop = EarlyStopping(monitor="val_loss", min_delta=0.001)
		# # Fit the model
		# self.__nn_classifier.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1, validation_split=0.05, callbacks=[early_stop])

		# retrain model using the optimum number of epochs
		self.__build_nn(num_features, [128, 64])
		# self.__nn_classifier.fit(X_train, y_train, nb_epoch=early_stop.stopped_epoch+1, batch_size=64, verbose=1)
		self.__nn_classifier.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1, shuffle=False)

		# extract features for the test set
		self.__feature_extractor.is_training = False
		[label_list, features_list, _] = self.__feature_extractor.extract_features(data_test, False, [], self.classifier)
		# number of data points in the test set
		num_datapoints = len(features_list)
		# build X matrix for input to the NN: X = num_datapoints x num_features
		X_test = np.zeros([num_datapoints, num_features])
		for i, features_dict in enumerate(features_list):
			for key, value in features_dict.items():
				# set values in the X matrix according to the dictionary
				X_test[i, key-1] = value
		# make predictions
		y_pred = self.__nn_classifier.predict(X_test)
		# restore/clear keras backend session
		backend.clear_session()
		# convert softmax output to labels [1,2,3] by taking the index of the max for each label (+1)
		y_pred = np.argmax(y_pred, axis=1) + 1
		# convert predictions to str
		y_pred = y_pred.astype(str)
		# convert labels from numbers to BIO
		y_pred[y_pred == "1"] = self.__feature_extractor.get_label_string(1)
		y_pred[y_pred == "2"] = self.__feature_extractor.get_label_string(2)
		y_pred[y_pred == "3"] = self.__feature_extractor.get_label_string(3)
		# convert to list
		y_pred = y_pred.tolist()
		# report performance
		if self.verbose:
			# parse predictions
			pred_labels = self.__parse_predictions(data_test, y_pred)
			# report performance
			self.__report_performance(data_test, pred_labels)

	def train(self, cv, data_train, data_test):
		"""
		Trains Minitagger on the given train data. If test data is given, it reports the accuracy of the trained model

		@type data_train: SequenceData
		@param data_train: the training data set
		@type data_test: SequenceData
		@param data_test: the test data set
		@type cv: boolean
		@type cv: flag to enable cross validation
		"""

		# training flag should be active for training
		assert (self.__feature_extractor.is_training), "In order to train, is_training flag should be True"
		# update class variable
		self.cross_val = cv
		# check if cross validation is enabled
		if cv:
			# number of cross-validation folds
			folds = 5
			# the training set contains the whole dataset when CV is used
			# take the lists of words and labels so that data can be split into train and test set
			data_set = data_train.sequence_pairs
			# cast the data set into np.array (necessary for using smart indexing and KFold)
			data_set = np.array(data_set)
			# initialize CRF object if CRF is used
			if self.classifier == "crf":
				self.__crf_classifier = sklearn_crfsuite.CRF(algorithm="lbfgs", all_possible_states=True, all_possible_transitions=True, c1=0.1, c3=0.1)
			# perform cross validation
			kf = KFold(n_splits=folds, random_state=6)
			# iterate through the data set and create train and test sets
			for fold, (train, test) in enumerate(kf.split(data_set)):
				# build two lists with targets and frequent words from the training set
				targets, frequent_words = build_name_lists(train)
				# equip feature extractor with target list and frequent words
				self.__feature_extractor.targets = targets
				self.__feature_extractor.frequent_words = frequent_words
				# create train data set
				data_train = data_set[train].tolist()
				# create SequenceData using data_train
				data_train = SequenceData(data_train)
				# create test data set
				data_test = data_set[test].tolist()
				# create SequenceData using the data_test
				data_test = SequenceData(data_test)
				print("\n----------------------------------")
				print(str(folds) + "-fold Cross-Validation: fold ", fold+1)
				print("----------------------------------\n")
				# fit right model and predict using the train and test sets respectively
				if self.classifier == "svm":
					self.__fit_and_predict_svm(data_train, data_test)
				if self.classifier == "crf":
					self.__fit_and_predict_crf(data_train, data_test)
				if self.classifier == "nn":
					self.__fit_and_predict_nn(data_train, data_test)
				# set training flag to true for the next iteration
				self.__feature_extractor.is_training = True
			# is_training is False after CV
			self.__feature_extractor.is_training = False
			assert(len(self.exact_fscore_list) == folds), "Length mismatch"
			assert(len(self.inexact_fscore_list) == folds), "Length mismatch"
			assert(len(self.conll_fscore_list) == folds), "Length mismatch"
			print("\nCross-validation finished:")
			print("\tExact f-scrore: {0:.3f}".format(np.mean(self.exact_fscore_list)))
			print("\tInexact f-scrore: {0:.3f}".format(np.mean(self.inexact_fscore_list)))
			print("\tCONLL f-scrore: {0:.3f}".format(np.mean(self.conll_fscore_list)))
			print()
			# log performance
			s = "{0:.3f}".format(np.mean(self.exact_fscore_list)) + "\t" + "{0:.3f}".format(np.mean(self.inexact_fscore_list)) + "\t" + "{0:.3f}".format(np.mean(self.conll_fscore_list)) + "\n"
			logger = open("results.csv", "a")
			logger.write(s)
			logger.close()
		else:
			# build two lists with targets and frequent words from the training set
			targets, frequent_words = build_name_lists(keep_all=True)
			# equip feature extractor with target list and frequent words
			self.__feature_extractor.targets = targets
			self.__feature_extractor.frequent_words = frequent_words
			# in case the CV is not enabled, use the given train and test data to fit and predict
			if self.classifier == "svm":
				self.__fit_and_predict_svm(data_train, data_test)
			if self.classifier == "crf":
				self.__crf_classifier = sklearn_crfsuite.CRF(algorithm="lbfgs", all_possible_states=True, all_possible_transitions=True, c1=0.1, c3=0.1)
				self.__fit_and_predict_crf(data_train, data_test)
			if self.classifier == "nn":
				self.__fit_and_predict_nn(data_train, data_test)

	def save(self, model_path):
		"""
		Saves the model as a directory at the given path

		@type model_path: str
		@param model_path: path to save the trained model
		"""

		# remove model_path if it already exists
		if os.path.exists(model_path):
			subprocess.check_output(["rm", "-rf", model_path])
		# make model_path directory
		os.makedirs(model_path)
		# save feature extractor in the model_path directory
		## if-else statement added on 06.02.2017
		if (self.__feature_extractor.feature_template == "relational") and (self.__feature_extractor.parser_type == "spacy"):
			print("Relational model with spaCy parser cannot be saved")
		else:
			pickle.dump(self.__feature_extractor, open(os.path.join(model_path, "feature_extractor"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
			# save trained model in the model_path directory
			liblinearutil.save_model(os.path.join(model_path, "liblinear_model"), self.__liblinear_model)

	def load(self, model_path):
		"""
		Loads the model from the directory at the given path

		@type model_path: str
		@param model_path: path to load the trained model
		"""

		# load feature_extractor object (used to extract features for the test set)
		## if-else statement added on 06.02.2017

		try:
			self.__feature_extractor = pickle.load(open(os.path.join(model_path, "feature_extractor"), "rb"))
			# load trained model
			self.__liblinear_model = liblinearutil.load_model(os.path.join(model_path, "liblinear_model"))
		except:
			raise Exception("No files found in the model path")

	def predict(self, data_test):
		"""
		Predicts tags in the given data
		It reports the accuracy if the data is fully labeled

		@type data_test: SequenceData
		@param data_test: the test data set
		@return: the predicted labels and the accuracy
		"""

		# keep starting timestamp
		start_time = time.time()
		assert (not self.__feature_extractor.is_training), "In order to predict, is_training should be False"

		# Extract features on all instances (labeled or unlabeled) of the test set
		[label_list, features_list, _] = self.__feature_extractor.extract_features(data_test, True, [], self.classifier)
		# pass them to liblinearutil for prediction
		pred_labels, (acc, _, _), _ = liblinearutil.predict(label_list, features_list, self.__liblinear_model, "-q")
		# print some useful information
		if not self.quiet:
			num_seconds = int(math.ceil(time.time() - start_time))
			# estimate prediction time
			print("Prediction time: {0}".format(str(datetime.timedelta(seconds=num_seconds))))
			# report accuracy if the data is fully labeled
			if not data_test.is_partially_labeled:
				print("Per-instance accuracy: {0:.3f}%".format(acc))
			else:
				print("Not reporting accuracy: test data is not fully labeled")

		# convert predicted labels from integer IDs to strings.
		for i, label in enumerate(pred_labels):
			pred_labels[i] = self.__feature_extractor.get_label_string(label)
		# create some files useful for debugging
		if self.verbose:
			# liblinear returns a list of tags
			# this list is parsed in order to create a list of lists of labels (each sublist contains the labels of a sentence)
			parsed_labels = self.__parse_predictions(data_test, pred_labels)
			# report performance
			self.__report_performance(data_test, parsed_labels)
		return pred_labels, acc

	def __parse_predictions(self, data_test, pred_labels):
		"""
		Parses the prediction labels of the liblinear

		@type data_test: SequenceData
		@param data_test: the test data set
		@type pred_labels: list
		@param data_test: list of the predicted labels
		@return: a list of lists of predicted labels
		each sublist contains labels for each sentence
		"""

		# list of parsed predictions
		predictions = []
		index = -1
		# iterate through the test data and parse predictions
		for words, labels in data_test.sequence_pairs:
			y_pred = []
			for word in words:
				index += 1
				# append label for each token
				y_pred.append(pred_labels[index])
			# append list of labels for each sentence
			predictions.append(y_pred)
		return predictions

	def __find_most_frequent_words(self, data_train):
		"""
		Computes the (active_seed_size) most frequent word types in data_train

		@type data_train: SequenceData
		@param data_train: t# Train for the first time.he training data set
		@return: the X most frequent words (X = active_seed_size) in train set
		"""

		# data_train.word_count is a dictionary with key = word , value = frequency of word in the train set
		# sort dictionary in descending order
		sorted_wordcount_pairs = sorted(data_train.word_count.items(),
										key=lambda type_count: type_count[1], reverse=True)
		# take the X most frequent words (X = active_seed_size)
		seed_wordtypes = [wordtype for wordtype, _ in sorted_wordcount_pairs[:self.active_seed_size]]
		return seed_wordtypes

	def __find_frequent_word_locations(self, data_train, seed_wordtypes):
		"""
		Finds random locations of the most frequent words

		@type data_train: SequenceData
		@param data_train: train data set
		@type seed_wordtypes: list
		@param seed_wordtypes: list of the X most frequent words (X = active_seed_size) in train set
		@return: a random location for each word in the seed_wordtypes
		each location is a tuple of the form (sequence number, position of the word in the sequence)
		"""

		occurring_locations = collections.defaultdict(list)
		# iterate through all sequences and words in the train set
		for sequence_num, (word_sequence, _) in enumerate(data_train.sequence_pairs):
			for position, word in enumerate(word_sequence):
				# append location is the current word is one of the seed_wordtypes
				if word in seed_wordtypes:
					occurring_locations[word].append((sequence_num, position))
		# take one random position for each word
		locations = [random.sample(occurring_locations[wordtype], 1)[0] for wordtype in seed_wordtypes]
		return locations

	def __make_data_from_locations(self, data_train, locations, skip_extraction):
		"""
		Makes SequenceData out of a subset of data_train from given location=(sequence_num, position) pairs

		@type data_train: SequenceData
		@param data_train: the train data set
		@type locations: list
		@param locations: list of tuples of locations for each one of the  X most frequent words
		(X = active_seed_size) in train set
		@return: a SequenceData object. It contains all words in the sequences. For words corresponding to locations,
		the true labels are used. Otherwise, the label is None
		"""

		# find all selected positions for each sequence
		selected_positions = collections.defaultdict(list)
		for (sequence_num, position) in locations:
			selected_positions[sequence_num].append(position)

		sequence_list = []
		for sequence_num in selected_positions:
			word_sequence, label_sequence = data_train.sequence_pairs[sequence_num]
			# initialize all labels to None
			selected_labels = [None for _ in range(len(word_sequence))]
			# take the right label for the words in the selected positions
			for position in selected_positions[sequence_num]:
				selected_labels[position] = label_sequence[position]
				# skip each word that corresponds to a position in the locations list
				# this example will not be selected again
				skip_extraction[sequence_num][position] = True
			# the sequence_list contains all words in a sequence. For words corresponding to locations, the true labels
			# are used. Otherwise, the label is None
			sequence_list.append((word_sequence, selected_labels))

		# make a SequenceData object using the sequence_list
		selected_data = SequenceData(sequence_list)
		return selected_data

	def __train_silently(self, data_selected):
		"""
		Trains on the selected data in silent mode

		@type data_selected: SequenceData
		@param data_selected: the selected subset of the training data set. Some words have correct labels
		and other words have None labels
		"""

		# reset for training
		self.__feature_extractor.is_training = True
		quiet_value = self.quiet
		self.quiet = True
		# no need for test set here.
		self.train(False, data_selected, None)
		self.quiet = quiet_value

	def __interval_report(self, data_selected, data_test, logfile):
		"""
		Reports accuracy in the specified interval

		@type data_selected: SequenceData
		@param data_selected: selected data based on locations
		@type data_test: SequenceData
		@param data_test: test data set
		@type logfile: file
		@param logfile: file used for log messages
		@return: None if it is not time to report output yet
		"""

		# report only at each interval
		if data_selected.num_labeled_words % self.active_output_interval != 0:
			return

		# test on the development data if any available
		if data_test is not None:
			quiet_value = self.quiet
			self.quiet = True
			# make prediction and return accuracy
			_, acc = self.predict(data_test)
			self.quiet = quiet_value
			message = "{0} labels: {1:.3f}%".format(data_selected.num_labeled_words, acc)
			print(message)
			logfile.write(message + "\n")
			logfile.flush()

		# Output the selected labeled examples so far.
		file_name = os.path.join(self.active_output_path, "example" + str(data_selected.num_labeled_words))
		with open(file_name, "w") as outfile:
			outfile.write(data_selected.__str__())

	def __find_confidence_index_pairs(self, confidence_index_pairs, scores_list):
		"""
		Estimates the confidence index pairs

		@type confidence_index_pairs:list
		@param confidence_index_pairs: list of tuples like (confidence, index)
		@type: list
		@param scores_list: list of scores for each word
		"""

		for index, scores in enumerate(scores_list):
			sorted_scores = sorted(scores, reverse=True)

			# handle the binary case
			# liblinear gives only 1 score whose sign indicates the class (+ versus -)
			confidence = sorted_scores[0] - sorted_scores[1] if len(scores) > 1 else abs(scores[0])
			confidence_index_pairs.append((confidence, index))

	def train_actively(self, data_train, data_test):
		"""
		Does margin-based active learning on the given data

		@type data_train: list
		@param data_train: list of training data set
		@type data_test: list
		@param data_test: list of test data set
		"""

		# for active learning, every data point (word) in the data_train should be labeled
		assert (not data_train.is_partially_labeled), "for active learning, every data point (word) " \
													  "in the data_train should be labeled"

		# keep track of which examples can be still selected for labeling
		__skip_extraction = []
		# initialize __skip_extraction to False for every label in every sequence
		# i.e. nothing is skipped at the beginning
		for _, label_sequence in data_train.sequence_pairs:
			__skip_extraction.append([False for _ in label_sequence])

		# create an output directory
		if os.path.exists(self.active_output_path):
			subprocess.check_output(["rm", "-rf", self.active_output_path])
		os.makedirs(self.active_output_path)
		logfile = open(os.path.join(self.active_output_path, "log"), "w")

		# take the X most frequent words (X = active_seed_size)
		seed_wordtypes = self.__find_most_frequent_words(data_train)
		# select a random location (sequence number, position in the sequence) of each selected type for a seed example
		locations = self.__find_frequent_word_locations(data_train, seed_wordtypes)
		# build a SequenceData object from the selected locations
		data_selected = self.__make_data_from_locations(data_train, locations, __skip_extraction)
		# train for the first time
		self.__train_silently(data_selected)
		self.__interval_report(data_selected, data_test, logfile)

		while len(locations) < data_train.num_labeled_words:
			# extract features for the remaining (i.e. not in the skip list) labeled examples
			[label_list, features_list, location_list] = self.__feature_extractor.extract_features(
				data_train, False, __skip_extraction, self.classifier)
			# make predictions on the remaining (i.e. not in the skip list) labeled examples
			_, _, scores_list = liblinearutil.predict(label_list, features_list, self.__liblinear_model, "-q")

			# compute "confidence" of each prediction:
			# max_{y} score(x,y) - max_{y'!=argmax_{y} score(x,y)} score(x,y')
			confidence_index_pairs = []
			self.__find_confidence_index_pairs(confidence_index_pairs, scores_list)

			# select least confident examples for next labeling
			confidence_index_pairs.sort()
			for _, index in confidence_index_pairs[:self.active_step_size]:
				locations.append(location_list[index])
			data_selected = self.__make_data_from_locations(data_train, locations, __skip_extraction)
			# train from scratch
			self.__train_silently(data_selected)
			self.__interval_report(data_selected, data_test, logfile)

		logfile.close()
