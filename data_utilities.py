from text_transformation import TextTransformation
import pandas as pd


class DataUtils:
    def __init__(self):
        self.transformer = TextTransformation()

    def preprocess(self, df):
        return self.transformer.transform_data(df)

    def get_train_data(self):
        train_data = pd.read_excel('data/train_nor_811.xlsx', engine="openpyxl")
        train_data = pd.DataFrame(train_data)
        return train_data

    def get_test_data(self):
        test_data = pd.read_excel('data/test_nor_811.xlsx', engine="openpyxl")
        test_data = pd.DataFrame(test_data)
        return test_data

    def get_valid_data(self):
        valid_data = pd.read_excel('data/valid_nor_811.xlsx', engine="openpyxl")
        valid_data = pd.DataFrame(valid_data)
        return valid_data

    def get_train_data_without_other(self):
        train_data = pd.read_excel('data/train_nor_811.xlsx', engine="openpyxl")
        train_data = pd.DataFrame(train_data)
        return self.transformer.remove_other_label(train_data)

    def get_test_data_without_other(self):
        test_data = pd.read_excel('data/test_nor_811.xlsx', engine="openpyxl")
        test_data = pd.DataFrame(test_data)
        return self.transformer.remove_other_label(test_data)

    def get_valid_data_without_other(self):
        valid_data = pd.read_excel('data/valid_nor_811.xlsx', engine="openpyxl")
        valid_data = pd.DataFrame(valid_data)
        return self.transformer.remove_other_label(valid_data)

    def export_excel(self, df, file_name):
        df.to_excel(file_name)
        print('Exported results to excel file named: ', file_name)


