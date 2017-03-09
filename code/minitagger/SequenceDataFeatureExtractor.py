import os
import sys
import math
import random
import pickle
import numpy as np
from utils import *
# from collections import Counter
from RelationalFeatureAnalyzer import RelationalFeatureAnalyzer


class SequenceDataFeatureExtractor(object):
	"""
	Extracts features from sequence data
	"""

	def __init__(self, feature_template, morphological_features, name_lists, language, parser_type):
		# dictionary with all morphological features
		self.morphological_feature_cache = {}
		# feature template used for feature extraction
		self.feature_template = feature_template
		# language of the data set
		self.language = language
		# parser type
		self.parser_type = parser_type
		# relational features with or without word embeddings
		self.enable_embeddings = False
		# path to data
		self.data_path = None
		# boolean flag for training
		self.is_training = True
		# use arc label
		self.arc_label = False
		# use arc head
		self.arc_head = False
		# use POS tag
		self.pos_tag = False
		# POS of the head
		self.head_pos = False
		# sequence of POS tags in a window around the word
		self.pos_window = False
		# flag to include morphological features or not
		self.include_morphological_features = morphological_features
		# use name lists
		self.use_name_lists = name_lists
		# dictionary that maps feature string to number
		self.__map_feature_str2num = {}
		# dictionary that maps feature number to string
		self.__map_feature_num2str = {}
		# dictionary that maps label string to number
		self.__map_label_str2num = {}
		# dictionary that maps label number to string
		self.__map_label_num2str = {}
		# dictionary that contains word embeddings (key = word, value = float array)
		self.__word_embeddings = None
		# dictionary that contains word bitstrings (key = word, value = bitstring)
		self.__word_bitstrings = None
		# symbol for unknown word that must be contained in the embeddings and bitstrings dictionaries
		self.unknown_symbol = "<?>"
		## added on 06.02.2017
		# initialize relational features analyzer
		if self.feature_template == "relational":
			self.relational_feature_analyzer = RelationalFeatureAnalyzer(self.parser_type, self.language)
		# initialize name lists
		self.targets = None
		self.frequent_words = None

	def num_feature_types(self):
		"""
		Finds the number of distinct feature types

		@return: the number of distinct feature types
		"""

		return len(self.__map_feature_str2num)

	def get_feature_string(self, feature_number):
		"""
		Converts a numeric feature ID to a string

		@type feature_number: int
		@param feature_number: numeric id of feature
		@return: given a feature number, it returns the respective feature string
		"""

		assert (feature_number in self.__map_feature_num2str), "Feature id not in featureID-to-string dictionary"
		return self.__map_feature_num2str[feature_number]

	def get_label_string(self, label_number):
		"""
		Converts a numeric label ID to a string

		@type label_number: int
		@param label_number: numeric id of id
		@return: the label string that corresponds to the given label number
		"""

		assert (label_number in self.__map_label_num2str), "Label id not in labelID-to-string dictionary"
		return self.__map_label_num2str[label_number]

	def get_feature_number(self, feature_string):
		"""
		Converts a feature string to a numeric ID

		@type feature_string: str
		@param feature_string: feature in string format
		@return: the numeric feature id given the feature string
		"""

		assert (feature_string in self.__map_feature_str2num), "Feature string not in featureString-to-ID dictionary"
		return self.__map_feature_str2num[feature_string]

	def get_label_number(self, label_string):
		"""
		Converts a label string to a numeric ID

		@type label_string: str
		@param label_string: label in string format
		@return: the numeric label id given the label string
		"""

		assert (label_string in self.__map_label_str2num), "Label string not in labelString-to-ID dictionary"
		return self.__map_label_str2num[label_string]

	def extract_features(self, sequence_data, extract_all, skip_list):
		"""
		Extracts features from the given sequence data.

		@type sequence_data: SequenceData object
		@param sequence_data: contains all word sequences and label sequences
		@type extract_all: bool
		@param extract_all: specifies if features should be extracted for all words or not.  Unless specified
		extract_all=True, it extracts features only from labeled instances
		@type skip_list: list
		@param skip_list: skips extracting features from examples specified by skip_list.
		This is used for active learning. (Pass [] to not skip any example.)
		@return: list of labels, list of features, list of locations (i.e. position in the corpus where each label is
		found)
		"""

		# list for labels
		label_list = []
		# list for features
		features_list = []
		# list for locations
		location_list = []
		# extract data path from sequence_data
		self.data_path = sequence_data.data_path
		# iterate through all sequences (=sentences) and all words in each sentence
		
		for sequence_num, (word_sequence, label_sequence) in enumerate(sequence_data.sequence_pairs):
			# check if relational features are used
			# if so, build relational info for the current word_sequence
			if self.feature_template == "relational":
				# get relational info for the current word_sequence
				relational_analysis = self.relational_feature_analyzer.analyze_relational_features(word_sequence)
			else:
				# relational info is not used
				relational_analysis = None
			for position, label in enumerate(label_sequence):
				# if this example is in the skip list, ignore.
				if skip_list and skip_list[sequence_num][position]:
					continue
				# only use labeled instances unless extract_all=True
				if (label is not None) or extract_all:
					# append label id to label list
					label_list.append(self.__get_label(label))
					# append feature id in features list
					features_list.append(self.__get_features(word_sequence, position, relational_analysis))
					# append location in locations list
					location_list.append((sequence_num, position))
		return label_list, features_list, location_list

	def __get_label(self, label):
		"""
		Finds the integer ID of the given label

		@type label: str
		@param label: label for which the label id is required
		@return: integer ID for given label
		"""

		if self.is_training:
			# if training, add unknown label types to the dictionary
			if label not in self.__map_label_str2num:
				# each time a new label arrives, the counter is incremented (start indexing from 1)
				label_number = len(self.__map_label_str2num) + 1
				# make the label string <--> id mapping in the dictionary
				self.__map_label_str2num[label] = label_number
				self.__map_label_num2str[label_number] = label
			# return label id
			return self.__map_label_str2num[label]
		else:
			# if predicting, take value from the trained dictionary
			if label in self.__map_label_str2num:
				return self.__map_label_str2num[label]
			# if label is not found, return -1
			else:
				return -1

	def __get_features(self, word_sequence, position, relational_analysis=None):
		"""
		Finds the integer IDs of the extracted features for a word at a given position in a sequence (=sentence)

		@type word_sequence: list
		@param word_sequence: sequence of words
		@type position: int
		@param position: position in the sequence of words
		@return: a dictionary of numeric features (key = feature id, value = value for the specific feature)
		"""

		# Extract raw features depending on the given feature template
		if self.feature_template == "baseline":
			raw_features = self.__get_baseline_features(word_sequence, position)
		elif self.feature_template == "embedding":
			assert (self.__word_embeddings is not None), "A path to embedding file should be given"
			raw_features = self.__get_embedding_features(word_sequence, position)
		elif self.feature_template == "bitstring":
			assert (self.__word_bitstrings is not None), "A path to bitstring file should be given"
			raw_features = self.__get_bitstring_features(word_sequence, position)
		## added on 06.02.2017
		## start
		elif self.feature_template == "relational":
			raw_features = self.__get_relational_features(word_sequence, position, relational_analysis)
		## end
		else:
			raise Exception("Unsupported feature template {0}".format(self.feature_template))
		# map extracted raw features to numeric
		numeric_features = self.__map_raw_to_numeric_features(raw_features)
		return numeric_features

	def __map_raw_to_numeric_features(self, raw_features):
		"""
		Maps raw features to numeric

		@type raw_features: dict
		@param raw_features: dictionary of raw features (key = feature string, value = feature value)
		@return: a numeric dictionary (key = feature id, value = feature value)
		"""

		numeric_features = {}
		# iterate through all given features
		for raw_feature in raw_features:
			if self.is_training:
				# if training, add unknown feature types to the dictionary
				if raw_feature not in self.__map_feature_str2num:
					# fix feature id for a given string feature
					# Note: Feature index has to starts from 1 in liblinear
					feature_number = len(self.__map_feature_str2num) + 1
					# do the mapping for the feature string <--> id
					self.__map_feature_str2num[raw_feature] = feature_number
					self.__map_feature_num2str[feature_number] = raw_feature
				# set value of the key=feature_id to the correct feature value from the raw_features dict
				numeric_features[self.__map_feature_str2num[raw_feature]] = raw_features[raw_feature]
			else:
				# if predicting, only consider known feature types.
				if raw_feature in self.__map_feature_str2num:
					numeric_features[self.__map_feature_str2num[raw_feature]] = raw_features[raw_feature]
		return numeric_features

	def load_word_embeddings(self, embedding_path, embedding_length):
		"""
		Loads word embeddings from a file in the given path

		@type embedding_path: str
		@param embedding_path: path to the file containing the word embeddings
		"""

		# load the word embeddings dictionary
		print("Loading word embeddings...")
		file_name = "word_embeddings_" + str(embedding_length) + ".p"
		file_name = os.path.join(embedding_path, file_name)
		self.__word_embeddings = pickle.load(open(file_name, "rb"))
		# the token for unknown word types must be present
		assert (self.unknown_symbol in self.__word_embeddings), "The token for unknown word types must be present in the embeddings file"

		# address some treebank token conventions.
		if "(" in self.__word_embeddings:
			self.__word_embeddings["-LCB-"] = self.__word_embeddings["("]
			self.__word_embeddings["-LRB-"] = self.__word_embeddings["("]
			self.__word_embeddings["*LCB*"] = self.__word_embeddings["("]
			self.__word_embeddings["*LRB*"] = self.__word_embeddings["("]
		if ")" in self.__word_embeddings:
			self.__word_embeddings["-RCB-"] = self.__word_embeddings[")"]
			self.__word_embeddings["-RRB-"] = self.__word_embeddings[")"]
			self.__word_embeddings["*RCB*"] = self.__word_embeddings[")"]
			self.__word_embeddings["*RRB*"] = self.__word_embeddings[")"]
		if "\"" in self.__word_embeddings:
			self.__word_embeddings["``"] = self.__word_embeddings["\""]
			self.__word_embeddings["''"] = self.__word_embeddings["\""]
			self.__word_embeddings["`"] = self.__word_embeddings["\""]
			self.__word_embeddings["'"] = self.__word_embeddings["\""]

	def load_word_bitstrings(self, bitstring_path):
		"""
		Loads word bitstrings from a file in the given path

		@type bitstring_path: str
		@param bitstring_path: path to file for bitstring
		"""

		self.__word_bitstrings = {}
		with open(bitstring_path, "r") as input_file:
			for line in input_file:
				tokens = line.split()
				if len(tokens) == 0:
					continue
				# the lines in the bitstring file look like
				# tokens = [bitstring] [type] [count]
				self.__word_bitstrings[tokens[1]] = tokens[0]

		# the token for unknown word types must be present.
		assert (self.unknown_symbol in self.__word_bitstrings), "The token for unknown word types must be present in the bitstring file"

		# address some treebank token replacement conventions.
		if "(" in self.__word_bitstrings:
			self.__word_bitstrings["-LCB-"] = self.__word_bitstrings["("]
			self.__word_bitstrings["-LRB-"] = self.__word_bitstrings["("]
			self.__word_bitstrings["*LCB*"] = self.__word_bitstrings["("]
			self.__word_bitstrings["*LRB*"] = self.__word_bitstrings["("]
		if ")" in self.__word_bitstrings:
			self.__word_bitstrings["-RCB-"] = self.__word_bitstrings[")"]
			self.__word_bitstrings["-RRB-"] = self.__word_bitstrings[")"]
			self.__word_bitstrings["*RCB*"] = self.__word_bitstrings[")"]
			self.__word_bitstrings["*RRB*"] = self.__word_bitstrings[")"]
		if "\"" in self.__word_bitstrings:
			self.__word_bitstrings["``"] = self.__word_bitstrings["\""]
			self.__word_bitstrings["''"] = self.__word_bitstrings["\""]
			self.__word_bitstrings["`"] = self.__word_bitstrings["\""]
			self.__word_bitstrings["'"] = self.__word_bitstrings["\""]

	def __morphological_features(self, word, relative_position):
		"""
		Extracts morphological features about the given word. Also, it considers the word's relative position.

		@type word: str
		@param word: given word to extract morphological features
		@type relative_position: int
		@param relative_position: relative position of word in the word sequence
		@return: a copy of the morphological_feature_cache for the specific (word, relative_position)
		"""
		
		if (word, relative_position) not in self.morphological_feature_cache:
			features = dict()
			# identify word
			features["word({0})={1}".format(relative_position, word)] = 1
			# check if word is capitalized
			features["is_capitalized({0})={1}".format(relative_position, is_capitalized(word))] = 1
			# build suffixes and preffixes for each word (up to a length of 4)
			for length in range(1, 5):
				features["prefix{0}({1})={2}".format(length, relative_position, get_prefix(word, length))] = 1
				features["suffix{0}({1})={2}".format(length, relative_position, get_suffix(word, length))] = 1
			# check if all chars are non-alphanumeric
			features["is_all_nonalphanumeric({0})={1}".format(relative_position, is_all_nonalphanumeric(word))] = 1
			# check if word can be converted to float, i.e. word is a number
			features["is_float({0})={1}".format(relative_position, is_float(word))] = 1
			self.morphological_feature_cache[(word, relative_position)] = features
		
		# Return a copy so that modifying that object doesn't modify the cache.
		return self.morphological_feature_cache[(word, relative_position)].copy()

	def __get_baseline_features(self, word_sequence, position):
		"""
		Builds the baseline features by using morphological of the word at the position
		and 2 words left and right of the word.

		@type word_sequence: list
		@param word_sequence: sequence of words
		@type position: int
		@param position: position of word in the given sequence
		@return: baseline features (dict)
		"""

		# get word at given position
		word = get_word(word_sequence, position)
		# get 2 words on the left and right
		word_left1 = get_word(word_sequence, position - 1)
		word_left2 = get_word(word_sequence, position - 2)
		word_right1 = get_word(word_sequence, position + 1)
		word_right2 = get_word(word_sequence, position + 2)

		# initialize features dictionary
		features = {}
		# check if morphological features should be used
		if self.include_morphological_features:
			# extract morphological features
			features = self.__morphological_features(word, 0)
		else:
			# add feature for the word itself
			features["word(0)={0}".format(word)] = 1	
		# add features for the words on the left and right side
		features["word(-1)={0}".format(word_left1)] = 1
		features["word(-2)={0}".format(word_left2)] = 1
		features["word(+1)={0}".format(word_right1)] = 1
		features["word(+2)={0}".format(word_right2)] = 1

		# # check if name lists should be taken into account
		if self.use_name_lists:
			assert self.targets is not None, "List of targets should not be empty"
			assert self.frequent_words is not None, "List of frequent words should not be empty"
			# check if word in present in the targets
			flag = (word.lower() in self.targets)
			# create respective features
			features["in_target={0}".format(flag)] = 1
			# check if word is present in the target tokens
			flag = (word.lower() in self.frequent_words)
			# create respective feature
			features["in_freq_words={0}".format(flag)] = 1
		return features

	def __get_word_embeddings(self, word_sequence, position, offset, features):
		"""
		Gets embeddings for a given word using the embeddings dictionary

		@type word_sequence: list
		@param word_sequence: sequence of words
		@type position: int
		@param position: position in the sequence of words
		@type offset: int
		@param offset: offset relative to position
		@type features: dict
		@param features: dictionary with features
		"""

		# get current word
		word = word_sequence[position + offset]
		# build offset string
		offset = str(offset) if offset <= 0 else "+" + str(offset)
		# get word embedding for the given word (all keys in the dictionary are in lower case)
		if word.lower() in self.__word_embeddings:
			word_embedding = self.__word_embeddings[word.lower()]
		else:
			word_embedding = self.__word_embeddings[self.unknown_symbol]
		# normalize vector
		word_embedding /= np.linalg.norm(word_embedding)
		# enrich given features dict
		for i, value in enumerate(word_embedding):
			features["embedding({0})_at({1})".format(offset, (i + 1))] = value

	def __get_embedding_features(self, word_sequence, position):
		"""
		Extract embedding features = normalized baseline features + (normalized) embeddings
		of current, left, and right words.

		@type word_sequence: list
		@param word_sequence: sequence of words
		@type position: int
		@param position: position in the sequence of words
		@return: full dict of features
		"""

		# compute the baseline feature vector and normalize its length to 1
		features = self.__get_baseline_features(word_sequence, position)
		# # assumes binary feature values
		# norm_features = math.sqrt(len(features))
		# # normalize
		# for feature in features:
		# 	features[feature] /= norm_features
		# extract word embedding for given and neighbor words
		self.__get_word_embeddings(word_sequence, position, 0, features)
		if position > 0:
			self.__get_word_embeddings(word_sequence, position, -1, features)
		if position < len(word_sequence) - 1:
			self.__get_word_embeddings(word_sequence, position, 1, features)
		return features

	def __get_word_bitstring(self, word_sequence, position, offset, features):
		"""
		Gets bitstring for a given word using the bitstrings dictionary

		@type word_sequence: list
		@param word_sequence: sequence of words
		@type position: int
		@param position: position in the sequence of words
		@type offset: int
		@param offset: offset relative to position
		@type features: dict
		@param features: dictionary with features
		"""

		# get current word
		word = word_sequence[position + offset]
		# build offset string
		offset = str(offset) if offset <= 0 else "+" + str(offset)
		# get bitstring for the given word
		if word in self.__word_bitstrings:
			word_bitstring = self.__word_bitstrings[word]
		else:
			word_bitstring = self.__word_bitstrings[self.unknown_symbol]
		# build also prefixes of the bitstring
		for i in range(1, len(word_bitstring) + 1):
			features["bitstring({0})_prefix({1})={2}".format(offset, i, word_bitstring[:i])] = 1
		features["bitstring({0})_all={1}".format(offset, word_bitstring)] = 1

	def __get_bitstring_features(self, word_sequence, position):
		"""
		Extract bitstring features = normalized baseline features + bitstring
		of current, left, and right words.

		@type word_sequence: list
		@param word_sequence: sequence of words
		@type position: int
		@param position: position in the sequence of words
		@return: full dict of features
		"""

		# compute the baseline feature vector and normalize its length to 1
		features = self.__get_baseline_features(word_sequence, position)
		# get bitstring for the given and neighbor words
		self.__get_word_bitstring(word_sequence, position, 0, features)
		if position > 0:
			self.__get_word_bitstring(word_sequence, position, -1, features)
		if position < len(word_sequence) - 1:
			self.__get_word_bitstring(word_sequence, position, 1, features)
		return features

	## added on 06.02.2017
	def __get_relational_features(self, word_sequence, position, relational_analysis):
		"""
		Extract relational features for each word in a sequence of words
		
		@type word_sequence: list
		@param word_sequence: list of words
		@type position: int
		@param position: position in the word of sequence
		@type relational_analysis: list
		@param relational_analysis: list of lists that contains all necessary relational information for each word
		in the word_sequence variable
		"""

		# build baseline features
		features = self.__get_baseline_features(word_sequence, position)
		# remove totally baseline features
		# features = {}

		if self.arc_label:
			# get the arc label from the relational_analysis list
			arc_label = relational_analysis[position][0]
			features["arc_label={0}".format(arc_label)] = 1
		if self.arc_head:
			# get the arc head from the relational_analysis list
			arc_head = relational_analysis[position][1]
			features["arc_head={0}".format(arc_head)] = 1
		if self.pos_tag:
			# get pos tag
			pos_tag = relational_analysis[position][2]
			features["pos_tag={0}".format(pos_tag)] = 1
		if self.head_pos:
			# get pos of tag
			head_pos = relational_analysis[position][3]
			features["head_pos={0}".format(head_pos)] = 1
		if self.pos_window:
			# get the POS tags from the previous and the next word
			indexes = np.array(range(-1,2)) + position
			pos_sequence = []
			for p in indexes:
				if p < 0:
					# out of bounds
					tag = "BEGIN"
				elif p >= len(relational_analysis):
					# out of bounds
					tag = "END"
				else:
					# get POS tag of adjacent word
					tag = relational_analysis[p][2]
				pos_sequence.append(tag)
			# create features from the adjacent POS tags
			features["pos_at(-1)={0}".format(pos_sequence[0])] = 1
			features["pos_at(+1)={0}".format(pos_sequence[2])] = 1
		
		# relational features enriched with word embeddings
		if self.enable_embeddings:
			# # assumes binary feature values
			# norm_features = math.sqrt(len(features))
			# # normalize
			# for feature in features:
			# 	features[feature] /= norm_features
			# extract word embedding for given and neighbor words
			self.__get_word_embeddings(word_sequence, position, 0, features)
			if position > 0:
				self.__get_word_embeddings(word_sequence, position, -1, features)
			if position < len(word_sequence) - 1:
				self.__get_word_embeddings(word_sequence, position, 1, features)
		return features
