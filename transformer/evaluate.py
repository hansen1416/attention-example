import os
import pickle
import random

import torch

from model import Transformer
from preprocessing import prepare_lang

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join("..", "data", "eng-deu.pkl"), "rb") as f:
    raw_data = pickle.load(f)

input_lang, output_lang, input_encoded, target_encoded = prepare_lang(raw_data)

print(input_lang.n_words, output_lang.n_words)

d_model = 512
num_heads = 8
num_layers = 6
d_ff = 1024
max_seq_length = 10
dropout = 0.1

transformer = Transformer(
    input_lang.n_words,
    output_lang.n_words,
    d_model,
    num_heads,
    num_layers,
    d_ff,
    max_seq_length,
    dropout,
)

transformer.to(device)

checkpoint_epoch = 40

# load the model
transformer.load_state_dict(
    torch.load(
        os.path.join("models", f"transformer_{checkpoint_epoch}.pth"),
        map_location=torch.device("cpu"),
    )
)

transformer.eval()

print("Model loaded successfully!")
# print(transformer)
