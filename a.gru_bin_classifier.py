import fasttext
import torch
import torch.nn as nn
import numpy as np
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ipdb import launch_ipdb_on_exception
from tqdm import tqdm
from functools import reduce
from redditors_comments_dataset import RedditComments


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EXPT_NAME = "default"
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 0.0001
HIDDEN_REPR_DIM = 200


class BinaryClassifier(nn.Module):
    def __init__(self, hidden_size, embedding_size=300):
        super(BinaryClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = fasttext.load_model("wiki.simple/wiki.simple.bin")
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 2)
        self.sm = nn.Softmax(dim=0)

    def forward(self, input, hidden):
        sequence_len = len(input)
        embedded_sequence = []
        for token in input:
            token_embedding = self.embedding.get_word_vector(token)
            embedded_sequence.append(torch.Tensor(token_embedding).float())

        embedded_sequence = torch.cat(embedded_sequence, 0)
        # GRU expects (seq_len, batch_size, input_size)
        embedded_sequence = embedded_sequence.view(sequence_len, -1, self.embedding_size)

        output = embedded_sequence
        output, hidden = self.gru(output, hidden)
        fc_out = self.fc(hidden)
        sm_out = self.sm(fc_out)

        return sm_out

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


if __name__ == "__main__":
    writer = SummaryWriter(log_dir=f'./log/{EXPT_NAME}')

    classifier = BinaryClassifier(
        hidden_size=HIDDEN_REPR_DIM,
        embedding_size=300
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    inital_hidden = classifier.initHidden()
    classifier.train()

    train = RedditComments(path_to_json="./data/anon_auth_dataset.json")
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE)

    eval = RedditComments(path_to_json="./data/anon_auth_dataset.json", train=False)
    eval_loader = torch.utils.data.DataLoader(eval, batch_size=BATCH_SIZE)

    for e in range(1, EPOCHS + 1):
        for X, y in train_loader:
            X = list(reduce(lambda x, y: x + y, X))
            y = torch.Tensor([0., 1.]) if y == 1 else torch.Tensor([1., 0.])
            y = y.view(1, 1, -1).float()
            y_hat = classifier.forward(input=X, hidden=inital_hidden)

            optimizer.zero_grad()
            train_loss = criterion(y_hat, y)
            train_loss.backward()
            optimizer.step()

        if e % 1 == 0:
            classifier.eval()

            agg_test_loss = 0
            for eval_X, eval_y in eval_loader:
                eval_X = list(reduce(lambda x, y: x + y, X))
                eval_y = torch.Tensor([0., 1.]) if y == 1 else torch.Tensor([1., 0.])
                eval_y = eval_y.view(1, 1, -1).float()
                pred_y = classifier.forward(input=X, hidden=inital_hidden)

                test_loss = criterion(pred_y, eval_y)
                agg_test_loss += test_loss.item()

            agg_test_loss /= len(eval)

            writer.add_scalars("LSTM_Classifier/loss", {
                "test": agg_test_loss,
                "train": train_loss.item()
            })

            classifier.train()
