from __future__ import unicode_literals, print_function, division
import time

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils import showPlot, timeSince
from preprocessing import SOS_token, MAX_LENGTH, get_dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # `input_size` is the number of words in the input language
        # `hidden_size` is the size of the hidden states
        # what `nn.Embedding` does is it creates a lookup table that maps each word to a vector of size `hidden_size`
        # if `hidden_size` is 128, then each word will be represented by a vector of size 128, which serve like a index to the word
        # `nn.Embedding` is like a linear layer that is used to map the indices of the words to their corresponding vectors
        self.embedding = nn.Embedding(input_size, hidden_size)
        # `nn.GRU` use Gated Mechanism, return output, hidden
        # output size is (batch_size, L, D∗H_out), when batch_first=True
        # where L is the length of the input sequence
        # D is the number of directions (1 for unidirectional, 2 for bidirectional)
        # H_out is the hidden size
        # hidden size is (D∗num_layers, batch_size, H_out)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        # `embedded` will have the shape of (`batch_size, MAX_LENGTH, hidden_size`)
        # where th3 3rd dimenison is like indices of the words
        embedded = self.dropout(self.embedding(input))
        # output shape is (batch_size, MAX_LENGTH, hidden_size)
        # hidden shape is (1, batch_size, hidden_size)
        output, hidden = self.gru(embedded)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # query shape is (batch_size, 1, hidden_size)
        # keys shape is (batch_size, MAX_LENGTH, hidden_size), `encoder_outputs`
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        # context shape is (batch_size, 1, hidden_size)
        # weights shape is (batch_size, 1, MAX_LENGTH)
        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        # `encoder_outputs` shape is (batch_size, MAX_LENGTH, hidden_size)
        # `encoder_hidden` shape is (1, batch_size, hidden_size)
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=device
        ).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            # decoder_output shape is (batch_size, 1, n_words in output_lang)
            # decoder_hidden shape is (1, batch_size, hidden_size)
            # attn_weights shape is (batch_size, 1, MAX_LENGTH)
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(
                    -1
                ).detach()  # detach from history as input

        # decoder_outputs shape is (batch_size, MAX_LENGTH, n_words in output_lang)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        # attentions shape is (batch_size, MAX_LENGTH, MAX_LENGTH), but it is not used
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        # input shape is (batch_size, 1)
        # hidden shape is (1, batch_size, hidden_size)
        # encoder_outputs shape is (batch_size, MAX_LENGTH, hidden_size)
        # embedded shape is (batch_size, 1, hidden_size)
        embedded = self.dropout(self.embedding(input))

        # query shape is (batch_size, 1, hidden_size)
        query = hidden.permute(1, 0, 2)

        # context shape is (batch_size, 1, hidden_size)
        # attn_weights shape is (batch_size, 1, MAX_LENGTH)
        context, attn_weights = self.attention(query, encoder_outputs)

        # input_gru shape is (batch_size, 1, 2 * hidden_size), concatenate `embedded` and `context`
        input_gru = torch.cat((embedded, context), dim=2)

        # output shape is (batch_size, 1, hidden_size)
        # hidden shape is (1, batch_size, hidden_size)
        output, hidden = self.gru(input_gru, hidden)

        # output shape is (batch_size, 1, n_words in output_lang)
        output = self.out(output)

        return output, hidden, attn_weights


def train_epoch(
    dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion
):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        # decoder_outputs.view(-1, decoder_outputs.size(-1)).shape) is of shape (batch_size * MAX_LENGTH, n_words in output_lang)
        # target_tensor.view(-1).shape is of shape (batch_size * MAX_LENGTH)
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train(
    train_dataloader,
    encoder,
    decoder,
    n_epochs,
    learning_rate=0.001,
    print_every=100,
    plot_every=100,
):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(
            train_dataloader,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
        )
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(
                "%s (%d %d%%) %.4f"
                % (
                    timeSince(start, epoch / n_epochs),
                    epoch,
                    epoch / n_epochs * 100,
                    print_loss_avg,
                )
            )

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


if __name__ == "__main__":

    hidden_size = 128
    batch_size = 32

    lang1 = "eng"
    lang2 = "deu"

    input_lang, output_lang, train_dataloader = get_dataloader(batch_size, lang1, lang2)

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)

    # save encoder and decoder to file
    torch.save(encoder.state_dict(), f"encoder_att_{lang1}_{lang2}.pth")
    torch.save(decoder.state_dict(), f"decoder_att_{lang1}_{lang2}.pth")
