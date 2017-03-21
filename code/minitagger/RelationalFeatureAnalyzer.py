import re
import sys
import spacy
import pickle


class RelationalFeatureAnalyzer(object):
	"""
	Performs relational feature analysis for a given sequence of words
	"""
	
	def __is_punctuation(self, token):
		"""
		Checks if a token is a punctuation symbol or not (apart from dot and comma)

		@type token: spacy object
		@param token: item form inspection
		@return: True if punctuation symbol (apart from comma and dot)
		"""

		if token.is_punct:
			if token.text == "." or token.text == ",":
				return False
			else:
				return True
		return False

	def __init__(self, parser_type, language):
		# define parser type, i.e. which parser to use
		self.parser_type = parser_type
		# define language to be used (language of the data set)
		self.language = language
		if self.parser_type == "spacy":
			print("Loading spacy...")
			self.nlp = spacy.load(self.language)
			print("Spacy loaded")
		elif self.parser_type == "stanford":
			print("Not supported at the moment")
			sys.exit()
		elif self.parser_type == "syntaxnet":
			print("Not supported at the moment")
			sys.exit()			
		else:
			raise Exception("Unsupported parser type {0}".format(self.parser_type))

	def __relational_features_analysis_spacy(self, word_sequence):
		"""
		Takes a list of words that represents a sentence and performs relational feature analysis using the spacy parser.
		Information about the arc label, the arc head, the entity id and the iob id of each word is extracted
		
		@type word_sequence: list
		@param word_sequence: sequence of words
		@return: a list of lists with relational features. The position in the list corresponds to the position
		of the word in the sentence. The inner list contains the relational information about each word
		"""

		# join words to re-create sentence
		sentence = " ".join(word_sequence)
		# pass sentence to nlp object
		doc = self.nlp(sentence)
		# create a list to store relational information
		relational_analysis = []
		# append relational information for each word in the sentence
		for token in doc:
			# punctuation should be skipped because Stanford parser does not 
			# handle it (the parsers should treat punctuation in the same way)
			if self.__is_punctuation(token):
				continue
			# the position in the list indicates the position of the word in the sentence
			# relational information is given in the following way
			# ['label of incoming arc' | 'token at the source of the arc' | 'entity type' | 'entity iob']
			# relational_analysis.append([token.dep_, token.head.text, token.ent_type, token.ent_iob_])
			relational_analysis.append([token.dep_, token.head.text, token.pos_, token.head.pos_])
		return relational_analysis

	def __relational_features_analysis_stanford(self, word_sequence):
		pass

	def __relational_features_analysis_syntaxnet(self, word_sequence):
		pass

	def analyze_relational_features(self, word_sequence):
		"""
		Analyses a sequence of words and extracts relational information

		@type word_sequence: list
		@param word_sequence: list of words
		"""

		# call the appropriate function to perform relational analysis based
		# on the type of parser
		if self.parser_type == "spacy":
			relational_analysis = self.__relational_features_analysis_spacy(word_sequence)
		if self.parser_type == "stanford":
			relational_analysis = self.__relational_features_analysis_stanford(word_sequence)
		if (self.parser_type == "syntaxnet"):
			relational_analysis = self.__relational_features_analysis_syntaxnet(word_sequence)
		# the length of the relational_analysis list and the word_sequence
		# must be the same (relational_analysis list should have one entry
		# for each word in the word_sequence list)
		assert len(relational_analysis) == len(word_sequence), "Length mismatch"
		return relational_analysis
