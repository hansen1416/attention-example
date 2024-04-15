import os
import pickle

import numpy as np
import torch


SOS_token = 0
EOS_token = 1

MAX_LENGTH = 10


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(" ")]


def prepare_lang(raw_data):
    input_lang = Lang("eng")
    output_lang = Lang("deu")

    for pair in raw_data:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    # return input_lang, output_lang
    n = len(raw_data)
    input_encoded = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_encoded = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    # print(input_encoded.shape, target_encoded.shape)

    for idx, (inp, tgt) in enumerate(raw_data):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_encoded[idx, : len(inp_ids)] = inp_ids
        target_encoded[idx, : len(tgt_ids)] = tgt_ids

    # print(input_encoded)
    # print(target_encoded)

    return input_lang, output_lang, input_encoded, target_encoded


if __name__ == "__main__":

    from torch.utils.data import TensorDataset, DataLoader, RandomSampler

    with open(os.path.join("..", "data", "eng-deu.pkl"), "rb") as f:
        raw_data = pickle.load(f)

    print(len(raw_data))
    print(raw_data[0])

    input_lang, output_lang, input_encoded, target_encoded = prepare_lang(raw_data)

    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    print(input_encoded.shape, target_encoded.shape)

    train_data = TensorDataset(
        torch.LongTensor(input_encoded), torch.LongTensor(target_encoded)
    )

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

    print(len(train_dataloader))

    sample_feature, sample_target = next(iter(train_dataloader))

    print(sample_feature.shape, sample_target.shape)
