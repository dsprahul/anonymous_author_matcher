import json
import torch
import torchtext
import fasttext
import numpy as np
from torchtext.data import get_tokenizer
from a_gru_bin_classifier import (
    BinaryClassifier,
    HIDDEN_REPR_DIM
)
from uuid import uuid4
from tqdm import tqdm


ModelClass = BinaryClassifier
DATASET_PATH = "./data/anon_auth_dataset.json"
MODEL_PERSIST_PATH = "./models/gru_v0.7_2.5k_model.weights"
SEP = "<SEP>"


def get_text_representation(text):
    tokens = tokenizer(text)
    input = tokens + [SEP] + tokens
    input_embeddings = torch.Tensor([[
        embedding.get_word_vector(token)
        for token in input
    ]])
    _, representation = model.forward(
        input=input_embeddings.unsqueeze(0),
        hidden=torch.Tensor(model.initHidden())
    )

    return representation


def make_authors_db(model, train_df):
    AUTHORS_DB = {}
    UUID_AUTHORS = {}

    for author, comments in tqdm(train_df.items()):

        representation = np.zeros((len(comments), HIDDEN_REPR_DIM))
        for cid, comment in enumerate(comments):
            comments[cid] = get_text_representation(text=comment)
        representation = representation.mean(axis=0)

        author_uuid = uuid4()
        AUTHORS_DB[author_uuid] = representation
        UUID_AUTHORS[author_uuid] = author

    return AUTHORS_DB, UUID_AUTHORS


def find_author_id(query, threshold=0.9):
    for aid, repr in AUTHORS_DB.items():
        if repr.dot(query) > threshold:
            return aid

    return uuid4()  # Return new ID


def determine_author(text):

    representation = get_text_representation(text)
    author_id = find_author_id(query=representation, threshold=0.9)

    if author_id in AUTHORS_DB:
        return {
            "id": author_id,
            "name": UUID_AUTHORS[author_id],
            "meta": "existing_author"
        }
    else:
        return {
            "id": author_id,
            "name": "<new_author>",
            "meta": "new_author"
        }


def evaluate(df, comments_per_author=20):
    accuracy = 0
    total_comments = len(df) * comments_per_author

    for author, comments in df.items():
        for comment in comments:
            predicted_author = determine_author(text=comment)
            accuracy += predicted_author == author

    return (100 * accuracy) / total_comments


if __name__ == "__main__":

    with open(DATASET_PATH) as in_:
        train_df, _, test_df = json.load(in_)
        print(f'training on {len(train_df)} authors...')

    model = ModelClass(hidden_size=HIDDEN_REPR_DIM)
    model.load_state_dict(
        torch.load(MODEL_PERSIST_PATH, map_location="cpu"),
        strict=False
    )

    print(f'Loading tokenizer & embedding model... ', end=' ')
    tokenizer = get_tokenizer("spacy")
    embedding = fasttext.load_model("wiki.simple/wiki.simple.bin")
    print(f'done.')

    print(f'Calculating authors train representations... ')
    AUTHORS_DB, UUID_AUTHORS = make_authors_db(
        model=model,
        train_df=train_df
    )

    # Do magic here
    train_accu = evaluate(df=train_df, comments_per_author=60)
    test_accu = evaluate(df=test_df, comments_per_author=20)
    print(f'Train accuracy: {train_accu:.2f}%, Test accuracy: {test_accu:.2f}%')
