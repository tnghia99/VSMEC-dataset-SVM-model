import pandas
from word_segmentation import WordSegmentation
import string
import re
import emoji



class TextTransformation(object):
    def __init__(self):
        self.tokenizer = WordSegmentation()

    def remove_last_num_3(self, text):
        if text.endswith('3'):
            text = text[:-1]
        return text

    def remove_special_char(self, text):
        return ''.join(u for u in text if u not in string.punctuation or u == "_")

    def transform_data(self, df):
        # word segmentation
        df.Sentence = df.Sentence.transform(lambda text: self.tokenizer.transform(text))
        # convert emoji to text
        df.Sentence = df.Sentence.transform(lambda text: emoji.demojize(text))
        # remove punctuation
        df.Sentence = df.Sentence.transform(lambda text:  self.remove_special_char(text))
        # remove text ends with 3
        df.Sentence = df.Sentence.transform(lambda text: self.remove_last_num_3(text))
        # remove multiple spaces
        df.Sentence = df.Sentence.transform(lambda text: re.sub(r'\s+', ' ', text, flags=re.I))
        return df

    def remove_other_label(self, df):
        # remove sentences were labeled 'Other'
        new_df = df.drop(df[df.Emotion == 'Other'].index)
        return new_df
    def print_transformed_data(self):
        df = pandas.read_excel('data/line.xlsx')
        print('RAW DATA:')
        print(df)
        print('______________________________________')
        # df = self.remove_other_label(df)
        # print('DATA AFTER REMOVED \'OTHER\' LABELED SENTENCES:')
        # print(df)
        # print('______________________________________')
        print('TRANSFORMED DATA:')
        print(self.transform_data(df))


if __name__ == '__main__':
    TextTransformation().print_transformed_data()
