import json
import torch
import fasttext
import torchtext
import random
import numpy as np
from torchtext.data import get_tokenizer
random.seed(42)


class RedditComments(torch.utils.data.Dataset):

    def __init__(self,
                 path_to_json,
                 num_samples=10000,
                 p2nr=0.5
                 ):

        with open(path_to_json) as in_:
            train_df, dev_df, test_df = json.load(in_)

        self.tokenizer = get_tokenizer("spacy")
        self.embedding = fasttext.load_model("wiki.simple/wiki.simple.bin")
        self.p2n_ratio = p2nr
        self.num_samples = num_samples
        self.SEP = self.embedding.get_word_vector("<SEP>")

        df = {}
        # merge train and dev into one & do this split using torch built-in
        for author in dev_df:
            df[author] = train_df[author] + dev_df[author]

        tokenized_df = self._preprocess(df)
        positives, negatives = self._example_mixer(tokenized_df)

        X = np.array(positives + negatives)
        y = np.array([1] * len(positives) + [0] * len(negatives))

        indexes = list(range(self.num_samples))
        random.shuffle(indexes)

        self.X = X[indexes]
        self.y = y[indexes]

    def _preprocess(self, df):

        tokenized_df = {}
        for auth, comments in df.items():

            tokenized_comments = [
                self.tokenizer(comment)
                for comment in comments
            ]

            tokenized_comment_embeddings = []
            for comment in tokenized_comments:

                comment_embeddings = [
                    self.embedding.get_word_vector(token)
                    for token in comment
                ]
                tokenized_comment_embeddings.append(comment_embeddings)

            tokenized_df[auth] = tokenized_comment_embeddings

        return tokenized_df

    def _example_mixer(self, df):

        authors = list(df.keys())

        positives = []
        negatives = []
        positives_count = int(self.num_samples * self.p2n_ratio)
        negatives_count = self.num_samples - positives_count

        for i in range(positives_count):
            i = i % len(authors)
            author = authors[i]
            author_comments = df[author]

            first, second = random.sample(author_comments, 2)
            positives.append(first + [self.SEP] + second)

        for i in range(negatives_count):
            i = i % len(authors) - 1
            author_a = authors[i]
            author_b = authors[i + 1]

            first = random.sample(df[author_a], 1)[0]
            second = random.sample(df[author_b], 1)[0]
            negatives.append(first + [self.SEP] + second)

        return positives, negatives

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return torch.Tensor(self.X[idx]), self.y[idx]
