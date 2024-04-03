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


def evaluateRandomly(encoder, decoder, n=10, lang1="eng", lang2="deu", sentence=""):

    input_lang, output_lang, pairs = prepareData(lang1, lang2, True)

    if sentence:

        # sentence to lower case
        sentence = sentence.lower()

        output_words, _ = evaluate(encoder, decoder, sentence, input_lang, output_lang)
        output_sentence = " ".join(output_words)
        print(">", sentence)
        print("<", output_sentence)
        return

    for i in range(n):
        pair = random.choice(pairs)
        print(">", pair[0])
        print("=", pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = " ".join(output_words)
        print("<", output_sentence)
        print("")


def evaluate_att(input_lang, output_lang, lang1="eng", lang2="deu", sentence=""):

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    # load encoder and decoder from file
    encoder.load_state_dict(torch.load(f"encoder_att_{lang1}_{lang2}.pth"))
    decoder.load_state_dict(torch.load(f"decoder_att_{lang1}_{lang2}.pth"))

    evaluateRandomly(encoder, decoder, lang1=lang1, lang2=lang2, sentence=sentence)


def evaluate_rnn(input_lang, output_lang, lang1="eng", lang2="deu", sentence=""):

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, output_lang.n_words).to(device)

    # load encoder and decoder from file
    encoder.load_state_dict(torch.load(f"encoder_rnn_{lang1}_{lang2}.pth"))
    decoder.load_state_dict(torch.load(f"decoder_rnn_{lang1}_{lang2}.pth"))

    evaluateRandomly(encoder, decoder, lang1=lang1, lang2=lang2, sentence=sentence)


if __name__ == "__main__":

    hidden_size = 128
    batch_size = 32

    lang1 = "eng"
    lang2 = "deu"

    input_lang, output_lang, _ = get_dataloader(batch_size, lang1, lang2)

    evaluate_att(
        input_lang,
        output_lang,
        lang1=lang1,
        lang2=lang2,
        sentence="es ist schon zehn uhr",
    )
    # evaluate_rnn(input_lang, output_lang)
