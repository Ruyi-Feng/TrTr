import torch
import torch.nn as nn
from trtr.layers import Decoder, DecoderLayer, Encoder, EncoderLayer


class Transformer(nn.Module):
    """
    Vanilla Transformer
    """
    def __init__(self, d_model,
                        n_heads,
                        num_encoder_layers,
                        num_decoder_layers,
                        activation,
                        dropout,
                        device):
        super(Transformer, self).__init__()

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MultiHeadAttentionLayer(d_model, n_heads, dropout),
                    d_model,
                    dropout=dropout,
                    activation=activation
                ) for l in range(num_encoder_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    MultiHeadAttentionLayer(d_model, n_heads, dropout),
                    MultiHeadAttentionLayer(d_model, n_heads, dropout),
                    d_model,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(num_decoder_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

    def forward(self, x_enc, x_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.encoder(x_enc, attn_mask=enc_self_mask)
        return self.decoder(x_dec, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)


class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()
        return embeddings


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_relative_position = 2

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, query, key, value, mask = None):
        #query = [batch, seq_len, d_model]
        #key = [batch, seq_len, d_model]
        #value = [batch, seq_len, d_model]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim = -1))

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2
        #x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()
        #x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.d_model)
        #x = [batch size, query len, hid dim]

        x = self.fc_o(x)
        #x = [batch size, query len, hid dim]

        return x
