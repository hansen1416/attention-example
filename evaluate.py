from __future__ import unicode_literals, print_function, division

import random
import torch

from preprocessing import EOS_token, prepareData, tensorFromSentence, get_dataloader
from train import EncoderRNN, AttnDecoderRNN
from train_rnn import DecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(
            encoder_outputs, encoder_hidden
        )

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


def evaluateRandomly(encoder, decoder, n=10):

    input_lang, output_lang, pairs = prepareData("eng", "fra", True)

    for i in range(n):
        pair = random.choice(pairs)
        print(">", pair[0])
        print("=", pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = " ".join(output_words)
        print("<", output_sentence)
        print("")


def evaluate_att(input_lang, output_lang):

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    # load encoder and decoder from file
    encoder.load_state_dict(torch.load("encoder_att.pth"))
    decoder.load_state_dict(torch.load("decoder_att.pth"))

    evaluateRandomly(encoder, decoder)


def evaluate_rnn(input_lang, output_lang):

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, output_lang.n_words).to(device)

    # load encoder and decoder from file
    encoder.load_state_dict(torch.load("encoder_rnn.pth"))
    decoder.load_state_dict(torch.load("decoder_rnn.pth"))

    evaluateRandomly(encoder, decoder)


if __name__ == "__main__":

    hidden_size = 128
    batch_size = 32

    input_lang, output_lang, _ = get_dataloader(batch_size)

    evaluate_att(input_lang, output_lang)
    # evaluate_rnn(input_lang, output_lang)
