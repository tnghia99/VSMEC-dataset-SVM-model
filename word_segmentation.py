from pyvi import ViTokenizer


class WordSegmentation:
    def __init__(self):
        self.tokenizer = ViTokenizer

    def transform(self, X):
        result = self.tokenizer.tokenize(X)
        return result


if __name__ == '__main__':
    sentence = 'Đây là công cụ tách từ'
    print(WordSegmentation().transform(sentence))
