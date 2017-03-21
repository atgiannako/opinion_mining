import pickle
import string
from collections import Counter

def extract_frequent_words(targets):
	translator = str.maketrans('', '', string.punctuation)
	frequent_words = Counter()
	for item in targets:
		for token in item.strip().split():
			w = token.strip().translate(translator)
			frequent_words[w.lower()] += 1
	frequent_words = [key for key in frequent_words.keys() if frequent_words[key] > 4]
	return frequent_words

def extract_targets(datapoints, keep_all):
	all_targets = pickle.load(open("../../data/name_lists.p", "rb"))
	targets = Counter()
	if not keep_all:
		for index in datapoints:
			for target in all_targets[index]:
				targets[target.lower()] += 1
	else:
		for _, tar in all_targets.items():
			for item in tar:
				targets[item.lower()] += 1
	targets = [key for key in targets.keys() if targets[key] > 2]
	return targets

def build_name_lists(datapoints=None, keep_all=False):
	targets = extract_targets(datapoints, keep_all)
	frequent_words = extract_frequent_words(targets)
	return targets, frequent_words