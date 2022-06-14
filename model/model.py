import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math


class FlakyDetect(nn.Module):
    def __init__(self, option):
        super(FlakyDetect, self).__init__()
        self.option = option
        self.lstm_layer = nn.LSTM(option.encode_size, option.hidden_dim, option.n_layers, batch_first=True)
        self.input_linear = nn.Linear(option.token_embed_size + option.method_embed_size, option.encode_size, bias=False)
        self.out_linear = nn.Linear(option.hidden_dim, option.test_vector_size, bias=False)
        self.classify_linear = nn.Linear(option.test_vector_size, option.label_count, bias=False)
        self.input_layer_norm = nn.LayerNorm(option.encode_size)
        self.softmax = F.softmax
        self.attention_parameter = Parameter(
            torch.nn.init.xavier_normal_(torch.zeros(option.hidden_dim, 1, dtype=torch.float32, requires_grad=True)).view(-1),
            requires_grad=True,
        )

        if 0.0 < option.dropout_prob < 1.0:
            self.input_dropout = nn.Dropout(p=option.dropout_prob)
        else:
            self.input_dropout = None

    def forward(self, inputs, path_lengths, path_nums):

        # FNN, Layer Normalization, tanh
        combined_vectors = self.input_linear(inputs)
        combined_vectors = self.input_layer_norm(combined_vectors)
        combined_vectors = torch.tanh(combined_vectors)

        # dropout
        if self.input_dropout is not None:
            combined_vectors = self.input_dropout(combined_vectors)

        # LSTM
        packed_paths = nn.utils.rnn.pack_padded_sequence(combined_vectors, path_lengths, enforce_sorted=False, batch_first=True)
        _, (h_n, _) = self.lstm_layer(packed_paths)
        out = h_n[-1, :, :].squeeze()  # extract the final hidden state

        # code vector & attention
        out = torch.split(out, split_size_or_sections=path_nums, dim=0)
        code_vectors = torch.stack([torch.mean(torch.mul(path_vectors, self.get_attention(path_vectors)), dim=0) for path_vectors in out], dim=0)
        code_vectors = self.out_linear(code_vectors)

        # output
        outputs = self.classify_linear(code_vectors)

        return outputs, code_vectors, self.attention_parameter

    def get_attention(self, vectors):
        """calculate the attention of the output vetors."""
        attn_param = self.attention_parameter.unsqueeze(0).expand_as(vectors)
        attn_ca = torch.sum(vectors * attn_param, dim=1)
        attention = self.softmax(attn_ca, dim=0).reshape(-1, 1)
        return attention
