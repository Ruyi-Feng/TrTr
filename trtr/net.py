import copy
import torch.nn as nn
import torch
from adapter.device import torch_npu
from trtr.backbone import Transformer as Rltv
from trtr.backbone import Decoderonly as GPT
from trtr.backbone import DecoderonlyLayer, Block

class NrmlEmbedding(nn.Module):
    def __init__(self, config):
        super(NrmlEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self._pos_embed = nn.Parameter(torch.rand(1, config.input_len+config.pred_len, config.d_model))  # 对于每个token的pos_embed是一样的
        self.input_len = config.input_len
        # cov input x: [batch, seq_len, c_in]
        self._token_embed = nn.Conv1d(in_channels=(config.c_in), out_channels=config.d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)

    def forward(self, x):
        # x: batch, seq_len, c_in
        return self._token_embed(x.permute(0, 2, 1)).permute(0, 2, 1) + self._pos_embed.data[:, :x.shape[1], :]


class RltvEmbedding(nn.Module):
    def __init__(self, config):
        super(RltvEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # cov input x: [batch, seq_len, c_in]
        self._token_embed = nn.Conv1d(in_channels=(config.c_in), out_channels=config.d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)

    def forward(self, x):
        # x: batch, seq_len, c_in
        return self._token_embed(x.permute(0, 2, 1)).permute(0, 2, 1)


class StackEmbedding(nn.Module):
    def __init__(self, config):
        super(StackEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self._token_embed = nn.Conv1d(in_channels=(config.c_in), out_channels=config.d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self._frm_embed = nn.Embedding(config.frm_embed, config.d_model, padding_idx=0)
        self._car_embed = nn.Embedding(config.id_embed, config.d_model, padding_idx=0)
        self._pos_embed = nn.Parameter(torch.rand(1, config.max_car_num * config.input_len, 1))

    def forward(self, x_group):
        # x: batch, seq_len, c_in
        x = x_group[:, :, 2:]
        f = x_group[:, :, 0]
        c = x_group[:, :, 1]
        c_n = self._car_embed(c.int())
        f_n = self._frm_embed(f.int())
        t_n = self._token_embed(x.permute(0, 2, 1)).transpose(1, 2)
        return t_n + f_n + c_n + self._pos_embed.data[:, :x_group.shape[1], :]


class Trtr(nn.Module):
    net_factory = {
        "histregression": [False, nn.Transformer, 'nrml'],
        "selfregression": [False, GPT, 'nrml'],
        "histseq2seq": [False, nn.Transformer, None],
        "predict": [False, nn.Transformer, None],
        "mask": [False, nn.Transformer, None],
        "pad_tail": [False, nn.Transformer, None],
        "perfix": [False, GPT, 'perfix'],
        "histlabel": [False, nn.Transformer, None],
    }

    def __init__(self, config):
        super(Trtr, self).__init__()
        dp = config.dropout
        net_cfg = {"d_model": config.d_model,
                   "nhead": config.n_heads,
                   "num_encoder_layers":config.e_layers,
                   "num_decoder_layers":config.d_layers,
                   "activation":config.activation,
                   "dropout":dp}
        if config.pos_emb == 'rltv':
            self.batch_first = True
            Embedding = RltvEmbedding
            Net = Rltv
            net_cfg.setdefault("max_relative_position", config.max_relative_position)
            self.msk_type = self.net_factory[config.architecture][2]
        elif config.pos_emb == 'nrml':
            self.batch_first, Net, self.msk_type = self.net_factory[config.architecture]
            if config.data_form == 'stack':
                Embedding = StackEmbedding
            else:
                Embedding = NrmlEmbedding
        self.pred_len = config.pred_len
        self.enc_embeding = Embedding(config)
        self.dec_embeding = Embedding(config)
        self.trtr = Net(**net_cfg)
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.d_reduction = nn.Conv1d(in_channels=config.d_model, out_channels=(config.c_in),
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.criterion = nn.MSELoss()

    def forward(self, enc_x, dec_x, gt_x):
        enc_token = self.enc_embeding(enc_x)
        dec_token = self.dec_embeding(dec_x)
        if self.msk_type == "nrml":
            tgt_msk = self.trtr.generate_square_subsequent_mask(sz=dec_token.size(1)).to(enc_token.device)
        elif self.msk_type == "perfix":
            tgt_msk = self.trtr.generate_perfix_subsequent_mask(sz=dec_token.size(1)).to(enc_token.device) ## ABSENT
        else:
            tgt_msk = None
        if self.batch_first:
            output = self.trtr(enc_token, dec_token, dec_self_mask=tgt_msk).permute(0, 2, 1)
        else:
            output = self.trtr(enc_token.permute(1, 0, 2), dec_token.permute(1, 0, 2), tgt_mask=tgt_msk).permute(1, 2, 0)
        outputs = self.d_reduction(output).permute(0, 2, 1)  # -> batch, seq_len, d_model
        outputs = outputs[:, -gt_x.shape[1]:, :]
        loss = self.criterion(outputs, gt_x)
        return outputs, loss


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
        if config.data_form == 'stack':
            self._enc_embed = StackEmbedding(config)
            self._dec_pos_embed = nn.Parameter(torch.rand(1, config.input_len * config.max_car_num, config.d_model))
        else:
            self._enc_embed= NrmlEmbedding(config)
            self._dec_pos_embed = nn.Parameter(torch.rand(1, config.input_len + config.pred_len, config.d_model))
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # cov input x: [batch, seq_len, c_in]
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
        # embed tokens
        x = self._enc_embed(x)

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
