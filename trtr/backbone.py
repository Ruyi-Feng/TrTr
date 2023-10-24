import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from adapter.device import torch_npu


class Transformer(nn.Module):
    def __init__(self, d_model,
                 nhead,
                 num_encoder_layers,
                 num_decoder_layers,
                 activation,
                 dropout,
                 max_relative_position=10):
        super(Transformer, self).__init__()

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MultiHeadAttentionLayer(d_model, nhead, dropout, max_relative_position),
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
                    MultiHeadAttentionLayer(d_model, nhead, dropout, max_relative_position),
                    MultiHeadAttentionLayer(d_model, nhead, dropout, max_relative_position),
                    d_model,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(num_decoder_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

    def generate_square_subsequent_mask(self, sz: int):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x_enc, x_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.encoder(x_enc, attn_mask=enc_self_mask)
        return self.decoder(x_dec, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

class Decoderonly(nn.Module):
    def __init__(self, d_model,
                 nhead,
                 num_decoder_layers,
                 activation,
                 dropout,
                 num_encoder_layers=0):
        super(Decoderonly, self).__init__()

        # Decoder
        self.decoder = Block(
            [
                DecoderonlyLayer(
                    nn.MultiheadAttention(d_model, nhead, dropout),
                    d_model,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(num_decoder_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

    def generate_square_subsequent_mask(self, sz: int):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_perfix_subsequent_mask(self, sz: int):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        print("there are blank remained in perfix mask!!!!!! please check backbone perfix mask")
        pass

    def forward(self, x_enc, x_dec, tgt_mask=None):
        return self.decoder(x_dec, x_mask=tgt_mask)

class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(
            torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        if torch_npu is not None:
            final_mat = torch.LongTensor(final_mat).npu()
            embeddings = self.embeddings_table[final_mat].npu()
        else:
            final_mat = torch.LongTensor(final_mat).cuda()
            embeddings = self.embeddings_table[final_mat].cuda()
        return embeddings


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout, max_rltv_pos=2):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.max_relative_position = max_rltv_pos

        self.relative_position_k = RelativePosition(
            self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(
            self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('scale', torch.sqrt(
            torch.FloatTensor([self.head_dim])))

    def forward(self, query, key, value, attn_mask=None):
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

        r_q1 = query.view(batch_size, -1, self.nhead,
                          self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.nhead,
                        self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))

        r_q2 = query.permute(1, 0, 2).contiguous().view(
            len_q, batch_size*self.nhead, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.nhead, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim=-1))

        # attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.nhead,
                          self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(
            len_q, batch_size*self.nhead, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(
            batch_size, self.nhead, len_q, self.head_dim)

        x = weight1 + weight2
        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.d_model)
        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)
        # x = [batch size, query len, hid dim]

        return x


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(
            conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x = attn_layer(x, mask=attn_mask)
                x = conv_layer(x)
            x = self.attn_layers[-1](x)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)

class DecoderonlyLayer(nn.Module):
    def __init__(self, self_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderonlyLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross=None, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x

class Block(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Block, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x

class MAE(nn.Module):
    def __init__(self, config):
        super(MAE, self).__init__()
        # Encoder-Decoder
        self.encoder = Block(
            [
                DecoderonlyLayer(
                    nn.MultiheadAttention(config.d_model, config.n_heads, config.dropout),
                    config.d_model,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for l in range(config.e_layers)
            ],
            norm_layer=nn.LayerNorm(config.d_model),
        )
        self.decoder = Block(
            [
                DecoderonlyLayer(
                    nn.MultiheadAttention(config.d_model, config.n_heads, config.dropout),
                    config.d_model,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for l in range(config.e_layers)
            ],
            norm_layer=nn.LayerNorm(config.d_model),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self._enc_pos_embed = nn.Parameter(torch.rand(1, config.input_len+config.pred_len, config.d_model))
        self._dec_pos_embed = nn.Parameter(torch.rand(1, config.input_len+config.pred_len, config.d_model))
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # cov input x: [batch, seq_len, c_in]
        self._token_embed = nn.Conv1d(in_channels=(config.c_in), out_channels=config.d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self._token_reduction = nn.Conv1d(in_channels=config.d_model, out_channels=(config.c_in),
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self._token_embed(x.permute(0, 2, 1)).permute(0, 2, 1)

        # add pos embed w/o cls token
        x = x + self._enc_pos_embed[:, :x.shape[1], :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        x = self.encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self._dec_pos_embed[:, :x.shape[1], :]
        x = self.decoder(x.permute(1, 0, 2)).permute(1, 2, 0)

        # predictor projection
        x = self._token_reduction(x).permute(0, 2, 1)

        return x

    def forward_loss(self, gt_x, pred, mask):
        """
        gt_x: [batch, seq_len, c_in]
        pred:  [batch, seq_len, c_in]
        mask:  [batch, seq_len]
        """
        loss = (pred - gt_x) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per token
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed token
        return loss

    def forward(self, e_, d_, gt_x, mask_ratio=0.75):
        enc_x = copy.deepcopy(gt_x)
        latent, mask, ids_restore = self.forward_encoder(enc_x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(gt_x, pred, mask)
        return pred, loss