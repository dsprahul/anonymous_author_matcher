import argparse
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


parser = argparse.ArgumentParser(description="Trains a GRU binary classifer")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser.add_argument("-n", "--name")
args = parser.parse_args()

EXPT_NAME = args.name or "default"
DATASET_SIZE = 20
BATCH_SIZE = 1
EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_REPR_DIM = 300
MODEL_PERSIST_PATH = f'./{EXPT_NAME}_model.weights'

print(f'Running {EXPT_NAME} on {device}...')


class BinaryClassifier(nn.Module):
    def __init__(self, hidden_size, embedding_size=300):
        super(BinaryClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 2)
        self.sm = nn.Softmax(dim=2)

    def forward(self, input, hidden):

        sequence_len = input.size()[1]

        # GRU expects (seq_len, batch_size, input_size)
        input = input.view(sequence_len, -1, self.embedding_size)

        # output = embedded_sequence
        output, hidden = self.gru(input, hidden)

        # self.representation = hidden

        fc_out = self.fc(hidden)
        sm_out = self.sm(fc_out)

        return sm_out, hidden.data.cpu().numpy()

    def initHidden(self):
        return np.zeros((1, 1, self.hidden_size))


if __name__ == "__main__":
    writer = SummaryWriter(log_dir=f'./log/{EXPT_NAME}')

    dataset = RedditComments(
        path_to_json="./data/anon_auth_dataset.json",
        num_samples=DATASET_SIZE,
        p2nr=0.5
    )

    train_eval_splits = (int(DATASET_SIZE * 0.8), int(DATASET_SIZE * 0.2))
    train, eval = torch.utils.data.random_split(dataset, train_eval_splits)

    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE)
    eval_loader = torch.utils.data.DataLoader(eval, batch_size=BATCH_SIZE)

    classifier = BinaryClassifier(
        hidden_size=HIDDEN_REPR_DIM,
        embedding_size=300
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    hidden = classifier.initHidden()  # Initialize hidden state
    classifier.train()

    for e in range(1, EPOCHS + 1):
        print(f'\nTraining Epoch#{e}')
        train_accu, test_accu = 0, 0

        for X, y in tqdm(train_loader):

            y = torch.Tensor([0., 1.]) if y == 1 else torch.Tensor([1., 0.])
            y = y.view(1, 1, -1).float().to(device)
            y_hat, hidden = classifier.forward(
                input=X.to(device),
                hidden=torch.Tensor(hidden).to(device)
            )
            optimizer.zero_grad()
            train_loss = criterion(y_hat, y)
            train_accu += (torch.argmax(y_hat) == torch.argmax(y)).item()
            train_loss.backward()
            optimizer.step()

        if e % 1 == 0:
            classifier.eval()

            agg_test_loss = 0
            for eval_X, eval_y in eval_loader:

                eval_y = torch.Tensor([0., 1.]) if eval_y == 1 else torch.Tensor([1., 0.])
                eval_y = eval_y.view(1, 1, -1).float().to(device)
                pred_y, _ = classifier.forward(
                    input=eval_X.to(device),
                    hidden=torch.Tensor(hidden).to(device)
                )

                test_loss = criterion(pred_y, eval_y)
                agg_test_loss += test_loss.item()
                test_accu += (torch.argmax(pred_y) == torch.argmax(eval_y)).item()

            agg_test_loss /= len(eval)
            test_accu /= len(eval)
            train_accu /= len(train)

            writer.add_scalars("LSTM_Classifier/loss", {
                "test": agg_test_loss,
                "train": train_loss.item()
            }, e)

            writer.add_scalars("LSTM_Classifier/accuracy", {
                "test": test_accu,
                "train": train_accu
            }, e)

            torch.save(classifier.state_dict(), MODEL_PERSIST_PATH)

            classifier.train()
