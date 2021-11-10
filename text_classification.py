from model_training import ModelTraining
import os.path
import pandas as pd
from sklearn.metrics import classification_report
import pickle

class TextClassification():
    def __init__(self):
        self.model = ModelTraining()

    def predict_dev_data(self):
        vectorizer_dir = 'trained_model/vectorizer.pickle'
        trained_model_dir = 'trained_model/trained_model.pickle'
        if not os.path.exists(vectorizer_dir) or not os.path.exists(trained_model_dir):
            self.model.train_model()
        valid_data = self.model.data_util.get_valid_data()
        tfidf = pickle.load(open("trained_model/vectorizer.pickle", "rb"))
        valid_data = self.model.data_util.preprocess(valid_data)
        x_test = tfidf.transform(valid_data.Sentence)
        y_test = valid_data.Emotion
        trained_model = pickle.load(open("trained_model/trained_model.pickle", "rb"))
        predict = classification_report(y_test, trained_model.predict(x_test))
        print(predict)
        # Save the result
        emotion_predict = trained_model.predict(x_test)
        result = self.model.data_util.get_valid_data()
        result['Emotion Predict'] = emotion_predict
        self.model.data_util.export_excel(result, 'output/predict_valid_set.xlsx')

    def predict_dev_data_without_other(self):
        vectorizer_dir = 'trained_without_other/vectorizer_without_other.pickle'
        trained_model_dir = 'trained_without_other/trained_model_without_other.pickle'
        if not os.path.exists(vectorizer_dir) or not os.path.exists(trained_model_dir):
            self.model.train_model_without_other()
        valid_data = self.model.data_util.get_valid_data_without_other()
        tfidf = pickle.load(open("trained_without_other/vectorizer_without_other.pickle", "rb"))
        valid_data = self.model.data_util.preprocess(valid_data)
        x_test = tfidf.transform(valid_data.Sentence)
        y_test = valid_data.Emotion
        trained_model = pickle.load(open("trained_without_other/trained_model_without_other.pickle", "rb"))
        predict = classification_report(y_test, trained_model.predict(x_test))
        print(predict)
        # Save the result
        emotion_predict = trained_model.predict(x_test)
        result = self.model.data_util.get_valid_data_without_other()
        result['Emotion Predict'] = emotion_predict
        self.model.data_util.export_excel(result, 'output/predict_valid_set_without_other.xlsx')

    def predict_test_data(self):
        vectorizer_dir = 'trained_model/vectorizer.pickle'
        trained_model_dir = 'trained_model/trained_model.pickle'
        if not os.path.exists(vectorizer_dir) or not os.path.exists(trained_model_dir):
            self.model.train_model()
        test_data = self.model.data_util.get_test_data()
        tfidf = pickle.load(open("trained_model/vectorizer.pickle", "rb"))
        test_data = self.model.data_util.preprocess(test_data)
        x_test = tfidf.transform(test_data.Sentence)
        y_test = test_data.Emotion
        trained_model = pickle.load(open("trained_model/trained_model.pickle", "rb"))
        predict = classification_report(y_test, trained_model.predict(x_test))
        print(predict)
        # Save the result
        emotion_predict = trained_model.predict(x_test)
        result = self.model.data_util.get_test_data()
        result['Emotion Predict'] = emotion_predict
        self.model.data_util.export_excel(result, 'output/predict_test_set.xlsx')

    def predict_test_data_without_other(self):
        vectorizer_dir = 'trained_without_other/vectorizer_without_other.pickle'
        trained_model_dir = 'trained_without_other/trained_model_without_other.pickle'
        if not os.path.exists(vectorizer_dir) or not os.path.exists(trained_model_dir):
            self.model.train_model_without_other()
        test_data = self.model.data_util.get_test_data_without_other()
        tfidf = pickle.load(open("trained_without_other/vectorizer_without_other.pickle", "rb"))
        test_data = self.model.data_util.preprocess(test_data)
        x_test = tfidf.transform(test_data.Sentence)
        y_test = test_data.Emotion
        trained_model = pickle.load(open("trained_without_other/trained_model_without_other.pickle", "rb"))
        predict = classification_report(y_test, trained_model.predict(x_test))
        print(predict)
        # Save the result
        emotion_predict = trained_model.predict(x_test)
        result = self.model.data_util.get_test_data_without_other()
        result['Emotion Predict'] = emotion_predict
        self.model.data_util.export_excel(result, 'output/predict_test_set_without_other.xlsx')

    def predict_sentence(self, sentence):
        vectorizer_dir = 'trained_without_other/vectorizer_without_other.pickle'
        trained_model_dir = 'trained_without_other/trained_model_without_other.pickle'
        if not os.path.exists(vectorizer_dir) or not os.path.exists(trained_model_dir):
            self.model.train_model_without_other()
        tfidf = pickle.load(open("trained_without_other/vectorizer_without_other.pickle", "rb"))
        trained_model = pickle.load(open("trained_without_other/trained_model_without_other.pickle", "rb"))
        df_sentence = pd.DataFrame({
            'Sentence': [sentence]
        })
        print('Sentence after preprocessing:')
        df_sentence = self.model.data_util.preprocess(df_sentence)
        print(df_sentence)
        x_sentence = tfidf.transform(df_sentence.Sentence)
        predict = trained_model.predict(x_sentence)
        print('Emotion prediction: ')
        print(predict)
    #Classify valid data
    def run(self):
        print('Using VALID SET for classification testing')
        while True:
            choice = int(
                input("Press 1: Classify original valid set | Press 2: Classify valid data without \'Other\' labels: "))
            if choice == 1:
                print('Classify valid set. Results:')
                self.predict_dev_data()
                break
            if choice == 2:
                print('Classify valid set without \'Other\' labels. Results:')
                self.predict_dev_data_without_other()
                break
            else:
                break

    def demo(self):
        print('Using TEST SET for classification testing')
        while True:
            choice = int(input(
                "Press 1: Classify original test set | Press 2: Classify test data without \'Other\' labels: "))
            if choice == 1:
                print('Classify test set. Results:')
                self.predict_test_data()
                break
            if choice == 2:
                print('Classify test set without \'Other\' labels. Results:')
                self.predict_test_data_without_other()
                break
            else:
                break
if __name__ == '__main__':
    TextClassification().run()