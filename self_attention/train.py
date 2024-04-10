# https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention

import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(46)

sentence = "Life is short, eat dessert first"

dc = {s: i for i, s in enumerate(sorted(sentence.replace(",", "").split()))}

print(dc)

sentence_int = torch.tensor([dc[s] for s in sentence.replace(",", "").split()])
print(sentence_int)

vocab_size = 50_000

# here use 3 as the embedding size, only for demonstration, in practice, use a larger embedding size
# eg. Llama 2 utilizes embedding sizes of 4,096
embed = torch.nn.Embedding(vocab_size, 3)
# torch.Size([6, 3])
embedded_sentence = embed(sentence_int).detach()

print(embedded_sentence.shape)


class SelfAttention(nn.Module):

    def __init__(self, d_in, d_out_kq, d_out_v):
        """
        Args:
            d_in: int, embedding_size, we use 3 as the embedding size, only for demonstration,
                in practice, use a larger embedding size. eg. Llama 2 utilizes embedding sizes of 4,096

            d_out_kq: int, the number of elements in the query and key vectors, d_q = d_k
                Since we are computing the dot-product between the query and key vectors,
                these two vectors have to contain the same number of elements (d_q = d_k) `d_out_kq`

            d_out_v: int, the number of elements in the value vector v(i),
                In many LLMs, we use the same size for the value vectors such that d_q = d_k = d_v.
                However, the number of elements in the value vector v(i),
                which determines the size of the resulting context vector, can be arbitrary.
        """
        super().__init__()
        self.d_out_kq = d_out_kq

        # (embedding_size, d_out_kq)
        self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        # (embedding_size, d_out_kq)
        self.W_key = nn.Parameter(torch.rand(d_in, d_out_kq))
        # (embedding_size, d_out_v)
        self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))

    def forward(self, x):
        # (sentence_length, embedding_size) @ (embedding_size, d_out_kq) = (sentence_length, d_out_kq)
        # each item in `keys` is the keys weights for each word in the sentence
        # represents what information each element in the sequence can provide.
        keys = x @ self.W_key
        # (sentence_length, embedding_size) @ (embedding_size, d_out_kq) = (sentence_length, d_out_kq)
        # each item in `queries` is the queries weights for each word in the sentence
        # represents what information a specific element in the sequence needs from others. therefore keys.T
        queries = x @ self.W_query
        # (sentence_length, embedding_size) @ (embedding_size, d_out_v) = (sentence_length, d_out_v)
        # each item in `values` is the values weights for each word in the sentence
        # holds the actual information of each element.
        values = x @ self.W_value

        # attention score $\omega_{i,j} = q^{(i)} k^{(j)}$
        # (sentence_length, d_out_kq) @ (d_out_kq, sentence_length) = (sentence_length, sentence_length)
        attn_scores = queries @ keys.T
        # to obtain the normalized attention weights, α (alpha),
        # by applying the softmax function. Additionally, 1/√{d_k} is used to scale $\omega$
        # before normalizing it through the softmax function
        # The scaling by d_k ensures that the Euclidean length of the weight vectors will be approximately in the same magnitude.
        # dim=-1. This ensures that the attention weights for each element (represented by rows in the tensor) sum up to 1.
        # (sentence_length, sentence_length)
        attn_weights = torch.softmax(attn_scores / self.d_out_kq**0.5, dim=-1)

        # the context vector z^(i), which is an attention-weighted version of our original query input x^(i),
        # including all the other input elements as its context via the attention weights:
        # (sentence_length, sentence_length) @ (sentence_length, d_out_v) = (sentence_length, d_out_v)
        context_vec = attn_weights @ values
        return context_vec


d_in, d_out_kq, d_out_v = 3, 2, 4

sa = SelfAttention(d_in, d_out_kq, d_out_v)

res = sa(embedded_sentence)

print(res, res.shape)


class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out_kq, d_out_v, num_heads):
        super().__init__()
        # each self-attention head will have its own set of weight matrices, they work in parallel
        self.heads = nn.ModuleList(
            [SelfAttention(d_in, d_out_kq, d_out_v) for _ in range(num_heads)]
        )

    def forward(self, x):
        # the shape of the output tensor is (sentence_length, num_heads * d_out_v)
        return torch.cat([head(x) for head in self.heads], dim=-1)


d_in, d_out_kq, d_out_v, num_heads = 3, 2, 4, 3

mha = MultiHeadAttentionWrapper(d_in, d_out_kq, d_out_v, num_heads)

res = mha(embedded_sentence)

print(res, res.shape)
