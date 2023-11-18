# Ahmet Serdar Çanak (1760039)

import matplotlib.pyplot as plt
from sklearn import datasets, svm
from random import randint

def opgave_1(digits, x_Training, y_Training, test_Indexes):
    for x in range(len(digits.data)):
        randomInt = randint(0, 1)
        if (randomInt == 0 and len(x_Training) <= (len(digits.data) / 3 * 2)) or len(test_Indexes) >= len(digits.data) / 3:
            x_Training.append(digits.data[x])
            y_Training.append(digits.target[x])
        elif (randomInt == 1 and len(test_Indexes) <= len(digits.data) / 3) or len(test_Indexes) >= (len(digits.data) / 3 * 2):
            test_Indexes.append(x)

def opgave_2(digits, clf, amountOfTests):
    # # De onderstaande for-loop print de estimation en het antwoord en plot ook de digit die daarbij hoort.
    # for item in range(amountOfTests):
    #     print("estimation", clf.predict(digits.data[item:item + 1]), "answer =", digits.target[item])
    #     plt.imshow(digits.images[item], cmap=plt.cm.gray_r, interpolation='nearest')
    #     plt.show()

    correctEstimations = 0

    for item in range(amountOfTests):
        if clf.predict(digits.data[item:item + 1]) == digits.target[item]:
            correctEstimations += 1

    print("Accuracy |", (correctEstimations / amountOfTests) * 100, "% |", amountOfTests, "Digits Tested |", amountOfTests - correctEstimations, "Mistakes |")

if __name__ == "__main__":
    digits = datasets.load_digits()
    clf = svm.SVC(gamma=0.001, C=100)

    x_Training, y_Training = [], []
    test_Indexes = []

    opgave_1(digits, x_Training, y_Training, test_Indexes)

    clf.fit(x_Training, y_Training)

    opgave_2(digits, clf, len(test_Indexes))