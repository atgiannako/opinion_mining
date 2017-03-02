import collections


class SequenceData(object):
	"""
	Represents a data set of sequences of words with labels
	The sequences can be partially labeled
	The data be loaded from a text file or a list
	"""

	def __init__(self, given_data):
		# path to the data
		self.data_path = None
		# number of words in the data
		self.num_of_words = 0
		# number of labeled words
		self.num_labeled_words = 0
		# boolean if the data set is partially labeled
		self.is_partially_labeled = False
		# contains sequence of: sentences (sequence of words) | sequence of labels
		self.sequence_pairs = []
		# dictionary for counting: key = label | value = number of occurrences for each label
		self.label_count = collections.Counter()
		# dictionary for counting: key = word | value = number of occurrences for each word
		self.word_count = collections.Counter()
		# dictionary for counting: key = word | value = dictionary for counting labels for each word
		# e.g. { "set" : {"verb":10, "noun":8} }
		# e.g. { "eat" : {"verb":15} }
		self.word_label_count = {}

		# check if given data is a path to file, i.e. a string
		if isinstance(given_data, str):
			# initialize sequence pairs from file
			self.__initialize_sequence_pairs_from_file(given_data)
		# check if given data is in form of a list
		elif isinstance(given_data, list):
			# initialize sequence pairs from list
			self.__initialize_sequence_pairs_from_list(given_data)
		# through exception if neither a file, nor a list are provided for given_data
		else:
			raise Exception("SequenceData can be initialized using either a string (file path) or a list")

		self.__initialize_sequencedata_attributes()

	def get_sequence_average_length(self):
		"""
		Calculates the average length of the sequences

		@return: average length of sequences (sentences) in the given data set
		"""

		length_sum = 0
		for word_sequence, _ in self.sequence_pairs:
			length_sum += len(word_sequence)
		return float(length_sum) / len(self.sequence_pairs)

	def __initialize_sequence_pairs_from_file(self, data_path):
		"""
		Initializes sequences from a text file
		The format of the file should be [word] [true label] [predicted label]
		True and predicted labels are optional
		Empty lines indicate sequence boundaries

		@type data_path: str
		@param data_path: the path to the data
		"""

		self.data_path = data_path
		with open(data_path, "r") as input_file:
			word_sequence = []
			label_sequence = []
			for line in input_file:
				tokens = line.split()
				assert (len(tokens) < 3), "Each line should contain no more than 3 columns"
				# if line is not empty
				if tokens:
					# the word is the 1st token
					word = tokens[0]
					# the label is the 2nd token, if it exists, otherwise label = None
					label = None if len(tokens) == 1 else tokens[1]
					if label is None:
						# set the partially labeled flag to True
						self.is_partially_labeled = True
					# build sequence of words
					word_sequence.append(word)
					# build sequence of labels
					label_sequence.append(label)
				# line is empty means that a new sentence if about to start
				else:
					# check if word_sequence is empty
					if word_sequence:
						# append word_sequence and label_sequence to sequence_pairs
						self.sequence_pairs.append([word_sequence, label_sequence])
						# flush lists for the next sentence
						word_sequence = []
						label_sequence = []

			# file may not end with empty line
			# data of last sentence should be also be appended to sequence_pairs
			if word_sequence:
				self.sequence_pairs.append([word_sequence, label_sequence])

	def __initialize_sequence_pairs_from_list(self, sequence_list):
		"""
		Initializes sequences from a given list
		Absent labels are denoted with None

		@type sequence_list: list
		@param sequence_list: the list contains sequence of words with their
		respective labels. Each element of the given list should have the
		following form:
		sequence_list[i] = [word_sequence, label_sequence]
		"""

		for sequence_pair in sequence_list:
			assert (len(sequence_pair) == 2), "Each sequence pair should have a length of 2"
			# sequence of words are located in sequence_pair[0]
			word_sequence = sequence_pair[0]
			# sequence of labels are located in sequence_pair[1]
			label_sequence = sequence_pair[1]
			assert (len(word_sequence) == len(label_sequence)), "Length of sequence of words and sequence of labels should be the same"
			# append data to sequence_pairs
			self.sequence_pairs.append([word_sequence, label_sequence])

	def __initialize_sequencedata_attributes(self):
		"""
		Initializes the SequenceData attributes from the loaded sequence_pairs list
		that is built either from file or list
		"""

		# iterate through all sentences and through all words|labels in the sequence_pairs list
		for word_sequence, label_sequence in self.sequence_pairs:
			for i in range(len(word_sequence)):
				# get each word in the word_sequence
				word = word_sequence[i]
				# increment num of words
				self.num_of_words += 1
				# count occurrences per word
				self.word_count[word] += 1
				# get label for the respective word
				label = label_sequence[i]
				# check if label is None
				if label is not None:
					# increment number of labeled words
					self.num_labeled_words += 1
					# count occurrences for each label
					self.label_count[label] += 1
					# put word in the word_label_count dict if not present
					if not (word in self.word_label_count):
						self.word_label_count[word] = collections.Counter()
						# increment counter for the word|label pair
						self.word_label_count[word][label] += 1

		# sort word_label_count[word] dictionary such that the most frequent label per word appears first
		# e.g. { "set" : {"verb":10, "noun":8} }
		for word in self.word_label_count:
			self.word_label_count[word] = sorted(self.word_label_count[word].items(),
												 key=lambda pair: pair[1], reverse=True)

	# overrides str method for SequenceData object
	def __str__(self):
		"""
		String representation of sequence pairs

		@return: a string that represents the SequenceData object
		"""
		
		string_rep = ""
		# build a string representation such that each line is of the form:
		# word_1	label_1
		# word_2
		# (empty line)
		# ...
		# word_n	label_n
		# the second column is empty if label is None
		# empty lines separate sentences
		for sequence_num, (word_sequence, label_sequence) in enumerate(self.sequence_pairs):
			for position in range(len(word_sequence)):
				string_rep += word_sequence[position]
				if not (label_sequence[position] is None):
					string_rep += "\t" + label_sequence[position]
				string_rep += "\n"
			if sequence_num < len(self.sequence_pairs) - 1:
				string_rep += "\n"
		return string_rep
