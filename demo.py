from text_classification import TextClassification

if __name__ == '__main__':
    print("Choose 1: test set result\t\t Choose 2: Predict sentence")
    while True:
        choice = int(input("Choose: "))
        if choice == 1:
            TextClassification().demo()
        elif choice == 2:
            sentence = input('Input sentence: ')
            TextClassification().predict_sentence(sentence)
        else:
            break






