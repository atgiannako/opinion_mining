import os
import argparse
from utils import analyze_data
from SequenceData import SequenceData
from SequenceDataFeatureExtractor import SequenceDataFeatureExtractor
from Minitagger import Minitagger

# Used for instances without gold labels
ABSENT_GOLD_LABEL = "<NO_GOLD_LABEL>"


def main(args):

	# if specified, just analyze the given data and return.
	# this data can be a prediction output file.
	if args.analyze:
		analyze_data(args.data_path)
		return

	# train or use a tagger model on the given data.
	minitagger = Minitagger()
	minitagger.classifier = args.classifier
	minitagger.quiet = args.quiet
	sequence_data = SequenceData(args.data_path)

	if args.enable_embeddings:
		assert args.embedding_path, "Embeddings path should be specified when embeddings are enabled"

	if args.feature_template == "relational":
		assert (args.arc_label or args.arc_head or args.pos_tag or args.head_pos or args.pos_window), "Cannot create relational features"

	if args.train:
		# initialize feature extractor with the right feature template
		feature_extractor = SequenceDataFeatureExtractor(args.feature_template, args.morphological_features, args.name_lists, args.language, args.parser_type, args.classifier)
		if args.feature_template == "relational":
			feature_extractor.enable_embeddings = args.enable_embeddings
			feature_extractor.arc_label = args.arc_label
			feature_extractor.arc_head = args.arc_head
			feature_extractor.pos_tag = args.pos_tag
			feature_extractor.head_pos = args.head_pos
			feature_extractor.pos_window = args.pos_window
		# load bitstring or embeddings data
		if args.embedding_path:
			feature_extractor.load_word_embeddings(args.embedding_path, args.embedding_length)
		if args.bitstring_path:
			feature_extractor.load_word_bitstrings(args.bitstring_path)
		# equip Minitagger with the appropriate feature extractor
		minitagger.equip_feature_extractor(feature_extractor)
		test_data = SequenceData(args.test_data_path) if args.test_data_path else None
		if test_data is not None:
			# Test data should be fully labeled
			assert (not test_data.is_partially_labeled), "Test data should be fully labeled"
		minitagger.verbose = args.verbose
		if minitagger.verbose:
			assert args.prediction_path, "Path for prediction should be specified"
			minitagger.prediction_path = args.prediction_path
		# normal training, no active learning used
		if not args.active:
			assert args.model_path
			minitagger.train(args.cv, sequence_data, test_data)
			# minitagger.save(args.model_path)
		# do active learning on the training data
		else:
			assert (args.active_output_path), "Active output path should not be empty"
			assert (args.classifier == "svm"), "Classifier should be SVM for active learning"
			# assign the right parameters to minitagger
			minitagger.active_output_path = args.active_output_path
			minitagger.active_seed_size = args.active_seed_size
			minitagger.active_step_size = args.active_step_size
			minitagger.active_output_interval = args.active_output_interval
			minitagger.train_actively(sequence_data, test_data)
	# predict labels in the given data.
	else:
		assert args.model_path
		minitagger.load(args.model_path)
		pred_labels, _ = minitagger.predict(sequence_data)

		# optional prediction output
		# write predictions to file
		if args.prediction_path:
			file_name = os.path.join(args.prediction_path, "predictions.txt")
			with open(file_name, "w") as outfile:
				label_index = 0
				for sequence_num, (word_sequence, label_sequence) in enumerate(sequence_data.sequence_pairs):
					for position, word in enumerate(word_sequence):
						if not label_sequence[position] is None:
							gold_label = label_sequence[position]
						else:
							gold_label = ABSENT_GOLD_LABEL
						outfile.write(word + "\t" + gold_label + "\t" + pred_labels[label_index] + "\n")
						label_index += 1
					if sequence_num < len(sequence_data.sequence_pairs) - 1:
						outfile.write("\n")

if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument("--data_path", type=str, help="path to data (used for training/testing)", required=True)
	argparser.add_argument("--analyze", action="store_true", help="Analyze given data and return")
	argparser.add_argument("--model_path", type=str, help="path to model directory", required=True)
	argparser.add_argument("--prediction_path", type=str, help="path to output data for prediction")
	argparser.add_argument("--train", action="store_true", help="train the tagger on the given data")
	argparser.add_argument("--feature_template", type=str, default="baseline",
						   help="feature template (default: %(default)s)")
	argparser.add_argument("--morphological_features", action="store_true", help="use morphological features")
	argparser.add_argument("--embedding_length", type=int, default=50, help="vector length for word embeddings", choices=[50, 100, 200, 300])
	argparser.add_argument("--embedding_path", type=str, help="path to word embeddings")
	argparser.add_argument("--bitstring_path", type=str, help="path to word bit strings (from a hierarchy of word types)")
	argparser.add_argument("--quiet", action="store_true", help="no messages")
	argparser.add_argument("--test_data_path", type=str, help="path to test data set (used for training)")
	argparser.add_argument("--active", action="store_true", help="perform active learning on the given data")
	argparser.add_argument("--active_output_path", type=str, help="path to output directory for active learning")
	argparser.add_argument("--active_seed_size", type=int, default=1,
						   help="number of seed examples for active learning (default: %(default)d)")
	argparser.add_argument("--active_step_size", type=int, default=1,
						   help="number of examples for labeling at each iteration in active learning (default: %(default)d)")
	argparser.add_argument("--active_output_interval", type=int, default=100,
						   help="output actively selected examples every time this value divides their number"
								"(default: %(default)d)")
	argparser.add_argument("--arc_label", action="store_true", help="use the arc label in relational features")
	argparser.add_argument("--arc_head", action="store_true", help="use the arc head in relational features")
	argparser.add_argument("--pos_tag", action="store_true", help="use the POS tag in relational features")
	argparser.add_argument("--head_pos", action="store_true", help="use the POS tag of the head in relational features")
	argparser.add_argument("--pos_window", action="store_true", help="take POS tag of previous and next word into account")
	argparser.add_argument("--language", type=str, choices=["en", "de"],
						   help="language of the data set [en, de]", required=True)
	argparser.add_argument("--parser_type", type=str, choices=["spacy", "stanford", "syntaxnet"], help="type of parser to be used for relational feature extraction [default = spacy]", default="spacy")
	argparser.add_argument("--enable_embeddings", action="store_true", help="enriches the relational feature space with word embeddings")
	argparser.add_argument("--name_lists", action="store_true", help="uses name lists obtained from the training data set")
	argparser.add_argument("--cv", action="store_true", help="use 10-fold cross-validation")
	argparser.add_argument("--verbose", action="store_true", help="produce some files for debugging and prints performance information")
	argparser.add_argument("--classifier", type=str, help="type of classifier to be used", choices=["svm", "crf", "nn"], required=True)
	parsed_args = argparser.parse_args()
	main(parsed_args)
