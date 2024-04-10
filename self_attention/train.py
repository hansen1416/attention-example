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

# print(embedded_sentence)
# # torch.Size([6, 3])
# print(embedded_sentence.shape)

# d is the embedding size, 3 in this case
d = embedded_sentence.shape[1]

# Since we are computing the dot-product between the query and key vectors,
# these two vectors have to contain the same number of elements (d_q = d_k)
d_q, d_k = 2, 2

# In many LLMs, we use the same size for the value vectors such that d_q = d_k = d_v.
# However, the number of elements in the value vector v(i),
# which determines the size of the resulting context vector, can be arbitrary.
d_v = 4

# (3, 2)
W_query = torch.nn.Parameter(torch.rand(d, d_q))
# (3, 2)
W_key = torch.nn.Parameter(torch.rand(d, d_k))
# (3, 4)
W_value = torch.nn.Parameter(torch.rand(d, d_v))


x_2 = embedded_sentence[1]
# (2,)
query_2 = x_2 @ W_query
# (2,)
key_2 = x_2 @ W_key
# (4,)
value_2 = x_2 @ W_value

queries = embedded_sentence @ W_query

print(queries.shape)

# (sentence_length, 2)
keys = embedded_sentence @ W_key
# (sentence_length, 4)
values = embedded_sentence @ W_value

# $\omega_{i,j} = q^{(i)} k^{(j)}$
# (sentence_length,)
omega = queries @ keys.T

print(omega, omega.shape)

# to obtain the normalized attention weights, α (alpha),
# by applying the softmax function. Additionally, 1/√{d_k} is used to scale $\omega$
# before normalizing it through the softmax function
# The scaling by d_k ensures that the Euclidean length of the weight vectors will be approximately in the same magnitude.
# (sentence_length,)
attention_weights = F.softmax(omega / d_k**0.5, dim=0)
print(attention_weights, attention_weights.shape)


#  the context vector z^(2), which is an attention-weighted version of our original query input x^(2),
# including all the other input elements as its context via the attention weights:
# (4,) d_v
context_vector_2 = attention_weights @ values

print(context_vector_2, context_vector_2.shape)


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
        keys = x @ self.W_key
        # (sentence_length, embedding_size) @ (embedding_size, d_out_kq) = (sentence_length, d_out_kq)
        # each item in `queries` is the queries weights for each word in the sentence
        queries = x @ self.W_query
        # (sentence_length, embedding_size) @ (embedding_size, d_out_v) = (sentence_length, d_out_v)
        # each item in `values` is the values weights for each word in the sentence
        values = x @ self.W_value

        # attention score $\omega_{i,j} = q^{(i)} k^{(j)}$
        # (sentence_length, d_out_kq) @ (d_out_kq, sentence_length) = (sentence_length, sentence_length)
        attn_scores = queries @ keys.T
        # to obtain the normalized attention weights, α (alpha),
        # by applying the softmax function. Additionally, 1/√{d_k} is used to scale $\omega$
        # before normalizing it through the softmax function
        # The scaling by d_k ensures that the Euclidean length of the weight vectors will be approximately in the same magnitude.
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
