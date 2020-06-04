import torch
import a_gru_bin_classifier
import torchtext
from torchtext.data import get_tokenizer
from uuid import uuid4

ModelClass = a_gru_bin_classifier.BinaryClassifier
MODEL_WEIGHTS = a_gru_bin_classifier.MODEL_PERSIST_PATH
HIDDEN_REPR_DIM = a_gru_bin_classifier.HIDDEN_REPR_DIM
SEP = "<SEP>"

model = ModelClass(hidden_size=HIDDEN_REPR_DIM)
model.load_state_dict(torch.load(MODEL_WEIGHTS))

model.eval()

tokenizer = get_tokenizer("spacy")

AUTHORS_DB = {}


def find_author_id(query, threshold=0.9):
    for aid, repr in AUTHORS_DB.items():
        if repr.dot(query) > threshold:
            return aid

    return uuid4()  # Return new ID


def get_text_representation(text):
    tokens = tokenizer(text)
    input = tokens + [SEP] + tokens
    model.forward(input=input, hidden=model.initHidden())

    return model.representation


def determine_author(text):

    representation = get_text_representation(text)
    author_id = find_author_id(query=representation, threshold=0.9)

    if author_id in AUTHORS_DB:
        return {
            "id": author_id,
            "meta": "existing_author"
        }
    else:
        return {
            "id": author_id,
            "meta": "new_author"
        }


if __name__ == "__main__":
    data_point = {
        "username": "<Name>",
        "source": "facebook",
        "body": "blah blah blah.."
    }
    data_point["author"] = determine_author(data_point["body"])

    db.collection.insert(
        data_point
    )
