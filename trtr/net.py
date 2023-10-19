import torch.nn as nn
import torch
from adapter.device import torch_npu
from trtr.backbone import Transformer as Rltv
from trtr.backbone import Decoderonly as GPT


class NrmlEmbedding(nn.Module):
    def __init__(self, config):
        super(NrmlEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self._pos_embed = nn.Parameter(torch.rand(1, config.input_len+config.pred_len, config.d_model))  # 对于每个token的pos_embed是一样的
        self.share = config.shared_pos_embed
        self.input_len = config.input_len
        # cov input x: [batch, seq_len, c_in]
        self._token_embed = nn.Conv1d(in_channels=(config.max_car_num*4), out_channels=config.d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)

    def forward(self, x, dec=False):
        # x: batch, seq_len, c_in
        if self.share or not dec:
            return self._token_embed(x.permute(0, 2, 1)).permute(0, 2, 1) + self._pos_embed.data[:, :x.shape[1], :]
        else:
            return self._token_embed(x.permute(0, 2, 1)).permute(0, 2, 1) + self._pos_embed.data[:, self.input_len:self.input_len+x.shape[1], :]


class RltvEmbedding(nn.Module):
    def __init__(self, config):
        super(RltvEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # cov input x: [batch, seq_len, c_in]
        self._token_embed = nn.Conv1d(in_channels=(config.max_car_num*4), out_channels=config.d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)

    def forward(self, x):
        # x: batch, seq_len, c_in
        return self._token_embed(x.permute(0, 2, 1)).permute(0, 2, 1)


class Trtr(nn.Module):
    net_factory = {
        "histregression": [False, nn.Transformer, 'nrml'],
        "selfregression": [False, GPT, 'nrml'],
        "histseq2seq": [False, nn.Transformer, None],
        "predict": [False, nn.Transformer, None],
        "mask": [False, nn.Transformer, None],
        "pad_tail": [False, nn.Transformer, None],
        "perfix": [False, GPT, 'perfix'],
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
        if config.model_type == 'rltv':
            self.batch_first = True
            Embedding = RltvEmbedding
            Net = Rltv
            self.msk_type = net_factory[self.config.architecture][2]
        elif config.model_type == 'nrml':
            self.batch_first, Net, self.msk_type = self.net_factory[config.architecture]
            Embedding = NrmlEmbedding
        self.pred_len = config.pred_len
        self.enc_embeding = Embedding(config)
        self.dec_embeding = Embedding(config)
        self.trtr = Net(**net_cfg)
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.d_reduction = nn.Conv1d(in_channels=config.d_model, out_channels=(config.max_car_num*4),
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.criterion = nn.MSELoss()

    def forward(self, enc_x, dec_x, gt_x):
        enc_token = self.enc_embeding(enc_x, dec=False)
        dec_token = self.dec_embeding(dec_x, dec=True)
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
