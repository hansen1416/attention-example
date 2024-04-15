import os
import pickle

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from model import Transformer
from preprocessing import prepare_lang


src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

transformer = Transformer(
    src_vocab_size,
    tgt_vocab_size,
    d_model,
    num_heads,
    num_layers,
    d_ff,
    max_seq_length,
    dropout,
)

# # Generate random sample data
# # (batch_size, seq_length)
# src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))
# # (batch_size, seq_length)
# tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))

# # print(src_data.size(), tgt_data.size())

with open(os.path.join("..", "data", "eng-deu.pkl"), "rb") as f:
    raw_data = pickle.load(f)

input_lang, output_lang, input_encoded, target_encoded = prepare_lang(raw_data)


train_data = TensorDataset(
    torch.LongTensor(input_encoded), torch.LongTensor(target_encoded)
)

train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)


criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in range(100):
    for input_tensor, target_tensor in train_dataloader:

        optimizer.zero_grad()
        # output = transformer(src_data, tgt_data[:, :-1])
        # loss = criterion(
        #     output.contiguous().view(-1, tgt_vocab_size),
        #     tgt_data[:, 1:].contiguous().view(-1),
        # )
        output = transformer(input_tensor, target_tensor)
        loss = criterion(
            output.contiguous().view(-1, tgt_vocab_size),
            target_tensor.contiguous().view(-1),
        )
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

        if epoch % 10 == 0:
            # save model to local file
            torch.save(transformer.state_dict(), f"transformer_{epoch}.pth")
