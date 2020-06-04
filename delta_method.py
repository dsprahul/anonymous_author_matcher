import numpy as np
import re
import string
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from function_words import fn_word_index

with open("./data/anon_auth_dataset.json") as in_:
    train, val, test = json.load(in_)


class CommentRepresentation(object):

    def __init__(self):
        self.model = np.zeros(len(fn_word_index))

    def transform(self, sentence):
        sent = re.sub('[%s]' % re.escape(string.punctuation), '', sentence)
        tokens = sent.lower().split()
        for tk in tokens:
            try:
                idx = fn_word_index[tk]
                self.model[idx] += 1
            except KeyError:
                continue

        return self.model

    def __add__(self, vector):
        return self.model + vector.model


def transform_dataset(dataset):

    train_X, train_y = [], []
    for auth, comments in dataset.items():

        vectorized_comments = []
        for comment in comments:
            x = CommentRepresentation().transform(sentence=comment)
            vectorized_comments.append(x)

        # vectorized_comments = MinMaxScaler().fit_transform(vectorized_comments)
        train_X += vectorized_comments
        train_y += [auth] * len(comments)

    return train_X, train_y


def train_KNN(train_ds):
    train_X, train_y = transform_dataset(train_ds)
    clf = KNeighborsClassifier(n_neighbors=100)
    clf.fit(train_X, train_y)

    return clf


def evaluate(test_ds, classifier):
    test_X, test_y = transform_dataset(test_ds)
    print(f'Accuracy: {clf.score(test_X, test_y):.2f}')


def make_model(dataset):

    train_X, train_y = [], []
    for auth, comments in dataset.items():

        vectorized_comments = []
        for comment in comments:
            x = CommentRepresentation().transform(sentence=comment)
            vectorized_comments.append(x)

        vectorized_comment = sum(vectorized_comments)
        train_X += [vectorized_comment]
        train_y += [auth]

    return train_X, train_y


def evaluate_model(dataset, model):
    train_X, train_y = model
    test_X, test_y = transform_dataset(dataset=dataset)

    predictions = []
    for X, y in zip(*[test_X, test_y]):

        min = float("inf")
        idx = 0
        for example in train_X:
            if example.dot(X) < min:
                min_idx = idx
            idx += 1

        pred_y = train_y[min_idx]
        predictions.append(y == pred_y)

    print(f'Accuracy {sum(predictions) / len(predictions):.2f}')


if __name__ == "__main__":

    # clf = train_KNN(train_ds=train)
    # evaluate(test_ds=test, classifier=clf)

    model = make_model(train)
    evaluate_model(dataset=test, model=model)
