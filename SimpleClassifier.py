"""
CreateAt: 24/01/2017
LastUpdatedAt: 06/02/2017
By: TH
"""
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from string import punctuation
import random
import numpy as np
from sklearn import cross_validation
import nltk.classify
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


class SimpleClassifier(object):

    def __init__(self, file):
        # Get dataframe
        self.dateframe = file
        # Get all the features
        self.get_all_feature_list()
        self.train_model()
        self.ten_fold_cross_validation()
        self.ten_fold_cross_validation_svm()
        self.ten_fold_cross_validation_logistic()
    
    def get_all_feature_list(self):
        """ Get training set """
        tweets = []
        print("training data contains {}".format(len(self.dateframe.index)))
        for row in range(len(self.dateframe.index)):
            tweet = self.pre_process(self.dateframe.iloc[row][0])
            category = self.dateframe.iloc[row][1]
            tweets.append((tweet, category))
        self.all_tweets = tweets
        self.word_features = self.get_word_features(self.get_words_in_tweets(self.all_tweets))

    @staticmethod
    def pre_process(tweet):
        """ pre process the tweets, including 1.remove stop words 2.tokenize 3.stemming"""
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(tweet)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        ps = PorterStemmer()
        feature_list = [ps.stem(w) for w in filtered_sentence if w not in punctuation]
        return feature_list

    @staticmethod
    def get_words_in_tweets(tweets):
        """ Get all the words from all the tweets"""
        all_words = []
        for (words, category) in tweets:
            for word in words:
                all_words.append(word.lower())
        return all_words

    @staticmethod
    def get_word_features(wordlist):
        """Count the number of times that each outcome of an experiment occurs"""
        wordlist = nltk.FreqDist(wordlist)
        #print(wordlist.most_common(50))
        word_features = wordlist.keys()
        return word_features
    
    def extract_features(self, tweet):
        """ extract the relevant features from the tweet """
        document_words = set(tweet)
        features = {}
        for word in self.word_features:
            features[word] = (word in document_words)
        return features

    def train_model(self):
        """ get the training set, but not does not train a model here."""
        training_set = nltk.classify.util.apply_features(self.extract_features, self.all_tweets)
        self.training_set = training_set
        #self.classifier = nltk.NaiveBayesClassifier.train(training_set)
    def ten_fold_cross_validation(self):
        """ 10 fold cross validation """
        self.cv = cross_validation.KFold(len(self.training_set), n_folds=10, shuffle=False, random_state=None)
        accuracy_sum = 0
        for traincv, testcv in self.cv:
            #classifier =  nltk.classify.SklearnClassifier(GaussianNB())
            #classifier.train(list(self.training_set[i] for i in traincv))
            classifier = nltk.NaiveBayesClassifier.train(list(self.training_set[i] for i in traincv))
            accuracy = nltk.classify.accuracy(classifier, list(self.training_set[i] for i in testcv))
            print('accuracy:', accuracy)
            accuracy_sum += accuracy
        print('average accuracy:', accuracy_sum/10)
    def ten_fold_cross_validation_svm(self):
        """ 10 fold cross validation """
        self.cv = cross_validation.KFold(len(self.training_set), n_folds=10, shuffle=False, random_state=None)
        accuracy_sum = 0
        for traincv, testcv in self.cv:
            classifier = nltk.classify.SklearnClassifier(LinearSVC())
            classifier.train(list(self.training_set[i] for i in traincv))
            accuracy = nltk.classify.accuracy(classifier, list(self.training_set[i] for i in testcv))
            print('accuracy:', accuracy)
            accuracy_sum += accuracy
        print('average accuracy:', accuracy_sum/10)

    def ten_fold_cross_validation_logistic(self):
        """ 10 fold cross validation """
        self.cv = cross_validation.KFold(len(self.training_set), n_folds=10, shuffle=False, random_state=None)
        accuracy_sum = 0
        for traincv, testcv in self.cv:
            classifier = nltk.classify.SklearnClassifier(LogisticRegression())
            classifier.train(list(self.training_set[i] for i in traincv))
            accuracy = nltk.classify.accuracy(classifier, list(self.training_set[i] for i in testcv))
            print('accuracy:', accuracy)
            accuracy_sum += accuracy
        print('average accuracy:', accuracy_sum/10)


if __name__ == "__main__":
    file = pd.read_csv('Python_Training.csv', sep=',', header=None)
    my_classifier = SimpleClassifier(file)

    



