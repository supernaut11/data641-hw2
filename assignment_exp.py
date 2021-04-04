import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import json
import gzip
import csv, re
import string
from tqdm import tqdm
import codecs
import argparse
from collections import Counter
from spacy.lang.en import English
from assignment1_fns import *

from experimental_features import RandomOversampler, LlrReduction

# Convenient for debugging but feel free to comment out
#from traceback_with_variables import activate_by_import

# Hard-wired variables
input_speechfile   = "./speeches2020_jan_to_jun.jsonl.gz"
stopwords_file     = "./mallet_en_stoplist.txt"
stopwords_file_jargon = "./new_legis_proc_jargon_stopwords.txt"

CLEANER_REGEX = re.compile(r'\s+')
MINIMUM_LLR_THRESHOLD = 25

# This is the similar to read_and_clean_lines in the previous assignment, but
# rather than just returning a list of cleaned lines of text, we should return
# returns two lists (of the same length): the cleaned lines and the party of the person who was speaking
#
# Read in congressional speeches jsonlines, i.e. a file with one well formed json element per line.
# Limiting to just speeches where the chamber was the Senate, return a list of strings
# in the following format:
#   '<party>TAB<text>'
# where <party> and <text> refer to the elements of those names in the json.
# Make sure to replace line-internal whitespace (newlines, tabs, etc.) in text with a space.
#
# For information on how to read from a gzipped file, rather than uncompressing and reading, see
# https://stackoverflow.com/questions/10566558/python-read-lines-from-compressed-text-files#30868178
#
# For info on parsing jsonlines, see https://www.geeksforgeeks.org/json-loads-in-python/.
# (There are other ways of doing it, of course.)
def read_and_clean_lines(infile, chamber):
    print("\nReading and cleaning text from {}".format(infile))
    lines = []
    parties = []

    with gzip.open(infile,'rt') as f:
        for line in tqdm(f):
            # Load the current json entry into a string
            cur_speech = json.loads(line)

            # Filter any results that weren't from the Senate
            if chamber == 'both' or cur_speech['chamber'].lower() == chamber:
                # Find text that needs to be cleaned and replace with with spaces, add
                # line to our list of speech data
                cleaned = CLEANER_REGEX.sub(' ', cur_speech['text'])
                lines.append(cleaned)
                parties.append(cur_speech['party'])

    print("Read {} documents".format(len(lines)))
    print("Read {} labels".format(len(parties)))
    return lines, parties

# Converting text into features.
# Inputs:
#    X - a sequence of raw text strings to be processed
#    analyzefn - either built-in (see CountVectorizer documentation), or a function we provide from strings to feature-lists
#
#    Arguments used by the words analyzer
#      stopwords - set of stopwords (used by "word" analyzer")
#      lowercase - true if normalizing by lowercasing
#      ngram_range - (N,M) for using ngrams of sizes N up to M as features, e.g. (1,2) for unigrams and bigrams
#
#  Outputs:
#     X_features - corresponding feature vector for each raw text item in X
#     training_vectorizer - vectorizer object that can now be applied to some new X', e.g. containing test texts
#    
# You can find documentation at https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# and there's a nice, readable discussion at https://medium.com/swlh/understanding-count-vectorizer-5dd71530c1b
#
def convert_text_into_features(X, stopwords_arg, analyzefn="word", range=(1,2)):
    training_vectorizer = CountVectorizer(stop_words=stopwords_arg,
                                          analyzer=analyzefn,
                                          lowercase=True,
                                          ngram_range=range)
    X_features = training_vectorizer.fit_transform(X)
    return X_features, training_vectorizer

# Input:
#    lines     - a raw text corpus, where each element in the list is a string
#    stopwords - a set of strings that are stopwords
#    remove_stopword_bigrams = True or False
#
# Output:  a corresponding list converting the raw strings to space-separated features
#
# The features extracted should include non-stopword, non-punctuation unigrams,
# plus the bigram features that were counted in collect_bigram_counts from the previous assignment
# represented as underscore_separated tokens.
# Example:
#   Input:  ["This is Remy's dinner.",
#            "Remy will eat it."]
#   Output: ["remy 's dinner remy_'s 's_dinner",
#            "remy eat"]
def convert_lines_to_feature_strings(lines, stopwords, remove_stopword_bigrams=True):

    print(" Converting from raw text to unigram and bigram features")
    if remove_stopword_bigrams:
        print(" Includes filtering stopword bigrams")
        
    print(" Initializing")
    nlp          = English(parser=False)
    all_features = []
    print(" Iterating through documents extracting unigram and bigram features")
    for line in tqdm(lines):
        
        # Get spacy tokenization and normalize the tokens
        spacy_analysis    = nlp(line)
        spacy_tokens      = [token.orth_ for token in spacy_analysis]
        normalized_tokens = normalize_tokens(spacy_tokens)

        # Collect unigram tokens as features
        # Exclude unigrams that are stopwords or are punctuation strings (e.g. '.' or ',')
        unigrams          = [token   for token in normalized_tokens
                                 if token not in stopwords and token not in string.punctuation]

        # Collect string bigram tokens as features
        bigrams           = ngrams(normalized_tokens, 2) 
        bigrams           = filter_punctuation_bigrams(bigrams)
        if remove_stopword_bigrams:
            bigrams = filter_stopword_bigrams(bigrams, stopwords)
        bigram_tokens = ["_".join(bigram) for bigram in bigrams]

        #trigrams = ngrams(normalized_tokens, 3)
        #trigrams = filter_punctuation_trigrams(trigrams)
        #if remove_stopword_bigrams:
        #    trigrams = filter_stopword_trigrams(trigrams, stopwords)
        #trigram_tokens = ["_".join(trigram) for trigram in trigrams]

        # Conjoin the feature lists and turn into a space-separated string of features.
        # E.g. if unigrams is ['coffee', 'cup'] and bigrams is ['coffee_cup', 'white_house']
        # then feature_string should be 'coffee cup coffee_cup white_house'
        feature_string = f"{' '.join(unigrams)} {' '.join(bigram_tokens)}"# {' '.join(trigram_tokens)}"

        # Add this feature string to the output
        all_features.append(feature_string)

    print(" Feature string for first document: '{}'".format(all_features[0]))
        
    return all_features

# For both classes, print the n most heavily weighted features in this classifier.
def most_informative_features(vectorizer, classifier, n=20):
    # Adapted from https://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers#11116960
    feature_names       = vectorizer.get_feature_names()
    coefs_with_features = sorted(zip(classifier.coef_[0], feature_names))
    top                 = zip(coefs_with_features[:n], coefs_with_features[:-(n + 1):-1])
    for (coef_1, feature_1), (coef_2, feature_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, feature_1, coef_2, feature_2))

# Split on whitespace, e.g. "a    b_c  d" returns tokens ['a','b_c','d']
def whitespace_tokenizer(line):
    return line.split()

class FeatureExtraction:
    def __init__(self):
        self._X_train = []
        self._X_test = []
        self._y_train = []
        self._y_test = []
        self._X_train_tokens = []
        self._X_test_tokens = []
        self._stopwords = set()

    # Read a set of stoplist words from filename, assuming it contains one word per line
    # Return a python Set data structure (https://www.w3schools.com/python/python_sets.asp)
    def load_stopwords(self, filename):
        stopwords = []
        with codecs.open(filename, 'r', encoding='ascii', errors='ignore') as fp:
            stopwords = fp.read().split('\n')
        self._stopwords |= set(stopwords)

        return self._stopwords
    
    def clear_stopwords(self):
        self._stopwords = set()
    
    # Call sklearn's train_test_split function to split the dataset into training items/labels
    # and test items/labels.  See https://realpython.com/train-test-split-python-data/
    # (or Google train_test_split) for how to make this call.
    #
    # Note that the train_test_split function returns four sequences: X_train, X_test, y_train, y_test
    # X_train and y_train  are the training items and labels, respectively
    # X_test  and y_test   are the test items and labels, respectively
    #
    # This function should return those four values
    def split_training_set(self, lines, labels, test_size=0.3, random_seed=42):
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(lines, labels, test_size=test_size,
            random_state=random_seed)

        print("Training set label counts: {}".format(Counter(self._y_train)))
        print("Test set     label counts: {}".format(Counter(self._y_test)))
        
        return self._X_train, self._X_test, self._y_train, self._y_test

    def tokenize_dataset(self):
        print("Creating feature strings for training data")
        self._X_train_tokens = convert_lines_to_feature_strings(self._X_train, self._stopwords)
        print("Creating feature strings for test data")
        self._X_test_tokens = convert_lines_to_feature_strings(self._X_test, self._stopwords)

    def _extract_features_sklearn(self):
        # Use sklearn CountVectorizer's built-in tokenization to get unigrams and bigrams as features
        X_features_train, training_vectorizer = convert_text_into_features(self._X_train, self._stopwords, "word", range=(1,2))
        
        return X_features_train, training_vectorizer, self._X_test

    def _extract_features_homebrew(self, llr_factor):
        X_train_feature_strings = self._X_train_tokens
        X_test_documents = self._X_test_tokens

        if llr_factor > 0:
            llr = LlrReduction(X_train_feature_strings, self._y_train, X_test_documents)
            X_train_feature_strings, X_test_documents = \
                llr.reduce_features(llr_factor, 'Democrat', 'Republican', MINIMUM_LLR_THRESHOLD)

        # Call CountVectorizer with whitespace-based tokenization as the analyzer, so that it uses exactly your features,
        # but without doing any of its own analysis/feature-extraction.
        X_features_train, training_vectorizer = convert_text_into_features(X_train_feature_strings, self._stopwords, whitespace_tokenizer)

        return X_features_train, training_vectorizer, X_test_documents

    def extract_features(self, llr_factor=0, use_sklearn=False):
        if not use_sklearn:
            return self._extract_features_homebrew(llr_factor)
        else:
            return self._extract_features_sklearn()

    def get_labels(self):
        return self._y_train, self._y_test

def prep_feature_extraction(chamber, filter_jargon):
    extractor = FeatureExtraction()

    extractor.load_stopwords(stopwords_file)
    if filter_jargon:
        extractor.load_stopwords(stopwords_file_jargon)

    # Read the dataset in and split it into training documents/labels (X) and test documents/labels (y)
    extractor.split_training_set(*read_and_clean_lines(input_speechfile, chamber))

    return extractor

class LogisticRegressionClassifier:
    def __init__(self):
        self._classifier = LogisticRegression(solver='liblinear', random_state=42)
        self._vectorizer = None

    def set_vectorizer(self, vectorizer):
        self._vectorizer = vectorizer

    def train(self, train, train_labels):
        self._classifier.fit(train, train_labels)

    def get_vectorized(self, data):
        return self._vectorizer.transform(data)

    def classify(self, test):
        # Apply the "vectorizer" created using the training data to the test documents, to create testset feature vectors
        X_test_features =  self._vectorizer.transform(test)

        # Classify the test data and see how well you perform
        # For various evaluation scores see https://scikit-learn.org/stable/modules/model_evaluation.html
        return self._classifier.predict(X_test_features)
    
    @property
    def classifier(self):
        return self._classifier

class LlrExperiment:
    def __init__(self, chamber, filter_jargon, oversample, output_handle):
        self._chamber = chamber
        self._filter_jargon = filter_jargon
        self._oversample = oversample
        self._output = output_handle

    def execute(self, llr_range):
        extractor = prep_feature_extraction(self._chamber, self._filter_jargon)

        y_train, y_test = extractor.get_labels()
        extractor.tokenize_dataset()

        lr_classifier = LogisticRegressionClassifier()

        self._output.write(','.join(['llr_value', 'accuracy', 'dem_precis', 'dem_recall', 'repub_precis', 'repub_recall']) + '\n')

        for i in llr_range:
            X_features_train, training_vectorizer, X_test_documents = extractor.extract_features(llr_factor=i)

                # If oversampling was requested, do the oversampling before classifying
            if self._oversample:
                X_features_train, y_train = RandomOversampler().fit_resample(X_features_train, y_train)

            # Create a logistic regression classifier trained on the featurized training data
            lr_classifier.set_vectorizer(training_vectorizer)
            lr_classifier.train(X_features_train, y_train)

            # Classify the test data and see how well you perform
            # For various evaluation scores see https://scikit-learn.org/stable/modules/model_evaluation.html
            print(f"Classifying test data for LLR factor = {i}")
            predicted_labels = lr_classifier.classify(X_test_documents)
            accuracy = metrics.accuracy_score(predicted_labels,  y_test)
            print(f'Accuracy  = {accuracy}')
            r_precision = metrics.precision_score(predicted_labels, y_test, pos_label='Republican')
            r_recall = metrics.recall_score(predicted_labels,    y_test, pos_label='Republican')
            d_precision = metrics.precision_score(predicted_labels, y_test, pos_label='Democrat')
            d_recall = metrics.recall_score(predicted_labels,    y_test, pos_label='Democrat')

            print(f'Precision for Republican = {r_precision}')
            print(f'Recall    for Republican = {r_recall}')
            print(f'Precision for Democrat   = {d_precision}')
            print(f'Recall    for Democrat   = {d_recall}')

            self._output.write(f'{i},{accuracy},{d_precision},{d_recall},{r_precision},{r_recall}\n')

def main(use_sklearn_feature_extraction, num_most_informative, plot_metrics, chamber='senate', filter_jargon=False, filter_by_llr=0,
    oversample=False):
    extractor = prep_feature_extraction(chamber, filter_jargon)
    y_train, y_test = extractor.get_labels()

    if use_sklearn_feature_extraction:
        # Use sklearn CountVectorizer's built-in tokenization to get unigrams and bigrams as features
        X_features_train, training_vectorizer, X_test_documents = extractor.extract_features(use_sklearn=True)
    else:
        # Call CountVectorizer with whitespace-based tokenization as the analyzer, so that it uses exactly your features,
        # but without doing any of its own analysis/feature-extraction.
        extractor.tokenize_dataset()
        X_features_train, training_vectorizer, X_test_documents = extractor.extract_features(llr_factor=filter_by_llr)

    # If oversampling was requested, do the oversampling before classifying
    if oversample:
        X_features_train, y_train = RandomOversampler().fit_resample(X_features_train, y_train)

    # Create a logistic regression classifier trained on the featurized training data
    lr_classifier = LogisticRegressionClassifier()
    lr_classifier.set_vectorizer(training_vectorizer)
    lr_classifier.train(X_features_train, y_train)

    # Show which features have the highest-value logistic regression coefficients
    print("Most informative features")
    most_informative_features(training_vectorizer, lr_classifier.classifier, num_most_informative)

    # Classify the test data and see how well you perform
    # For various evaluation scores see https://scikit-learn.org/stable/modules/model_evaluation.html
    print("Classifying test data")
    predicted_labels = lr_classifier.classify(X_test_documents)
    print('Accuracy  = {}'.format(metrics.accuracy_score(predicted_labels,  y_test)))
    for label in ['Republican', 'Democrat']:
        print('Precision for label {} = {}'.format(label, metrics.precision_score(predicted_labels, y_test, pos_label=label)))
        print('Recall    for label {} = {}'.format(label, metrics.recall_score(predicted_labels,    y_test, pos_label=label)))
    
    if plot_metrics:
        print("Generating plots")
        metrics.plot_confusion_matrix(lr_classifier.classifier, lr_classifier.get_vectorized(X_test_documents), y_test, normalize='true')
        plt.savefig('conf_matrix.png')
        metrics.plot_roc_curve(lr_classifier.classifier, lr_classifier.get_vectorized(X_test_documents), y_test)
        plt.show()
        plt.savefig('roc.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options for running this script')
    parser.add_argument('--use_sklearn_features', default=False, action='store_true', help="Use sklearn's feature extraction")
    parser.add_argument('--plot_metrics', default=False, action='store_true', help="Generate figures for evaluation")
    parser.add_argument('--num_most_informative', default=10, action='store', help="Number of most-informative features to show")
    parser.add_argument('--chamber', default='senate', type=lambda arg: arg.lower(), choices=['senate', 'house', 'both'], help='Chambers of Congress to include')
    parser.add_argument('--filter-jargon', default=False, action='store_true', help='Filter procedural jargon via stopwords')
    parser.add_argument('--filter-by-llr', default='0', type=int, help='Filter uni/bigrams by LLR importance (n most important)')
    parser.add_argument('--oversample', default=False, action='store_true', help='Oversample minority party samples')
    args = parser.parse_args()
    main(args.use_sklearn_features, int(args.num_most_informative), args.plot_metrics, args.chamber, args.filter_jargon, args.filter_by_llr, args.oversample)