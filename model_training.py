from data_utilities import DataUtils
import os.path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import pickle


class ModelTraining(object):
    def __init__(self):
        self.data_util = DataUtils()
    ###best params {'C': 10, 'gamma': 1, 'kernel': 'rbf'}
    def train_model(self):
        data = self.data_util.get_train_data()
        data = self.data_util.preprocess(data)
        # exported processed data to excel
        self.data_util.export_excel(data, 'output/check_original.xlsx')
        # train tfidf model
        tfidf = TfidfVectorizer()
        tfidf.fit(data.Sentence)
        # save vectorized result
        pickle.dump(tfidf, open("trained_model/vectorizer.pickle", "wb"))
        # train and save MLP model
        x_train = tfidf.transform(data.Sentence)
        y_train = data.Emotion
        model = svm.SVC(C=10, gamma=1, kernel='rbf')
        model.fit(x_train, y_train)
        # save MLP trained model
        pickle.dump(model, open("trained_model/trained_model.pickle", "wb"))

    def train_model_without_other(self):
        data = self.data_util.get_train_data_without_other()
        data = self.data_util.preprocess(data)
        # exported processed data to excel
        self.data_util.export_excel(data, 'output/check_without_other.xlsx')
        # train tfidf model
        tfidf = TfidfVectorizer()
        tfidf.fit(data.Sentence)
        # save vectorized result
        pickle.dump(tfidf, open("trained_without_other/vectorizer_without_other.pickle", "wb"))
        # train and save MLP model
        x_train = tfidf.transform(data.Sentence)
        y_train = data.Emotion
        model = svm.SVC(C=10, gamma=1, kernel='rbf')
        model.fit(x_train, y_train)
        # save MLP trained model
        pickle.dump(model, open("trained_without_other/trained_model_without_other.pickle", "wb"))

    def run(self):
        while True:
            choice = int(input("Press 1: original data |  Press 2: removed other labeled sentences:  "))
            if choice == 1:
                print('Train original data')
                vectorizer_dir = 'trained_model/vectorizer.pickle'
                trained_model_dir = 'trained_model/trained_model.pickle'
                if not os.path.exists(vectorizer_dir) or not os.path.exists(trained_model_dir):
                    self.train_model()
                else:
                    print('Model was trained.')
                break
            if choice == 2:
                print('Train data after removing labeled sentences')
                vectorizer_dir = 'trained_without_other/vectorizer_without_other.pickle'
                trained_model_dir = 'trained_without_other/trained_model_without_other.pickle'
                if not os.path.exists(vectorizer_dir) or not os.path.exists(trained_model_dir):
                    self.train_model_without_other()
                else:
                    print('Model was trained.')
                break
            else:
                break

