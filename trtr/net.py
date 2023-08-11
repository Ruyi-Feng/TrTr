import torch.nn as nn
import torch
import torch_npu
from trtr.backbone import Transformer as Rltv


class NrmlEmbedding(nn.Module):
    def __init__(self, config):
        super(NrmlEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self._pos_embed = nn.Parameter(torch.rand(1, config.input_len, 1))  # 对于每个token的pos_embed是一样的
        # cov input x: [batch, seq_len, c_in]
        self._token_embed = nn.Conv1d(in_channels=(config.max_car_num*4), out_channels=config.d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)

    def forward(self, x):
        # x: batch, seq_len, c_in
        return self._token_embed(x.permute(0, 2, 1)).permute(0, 2, 1) + self._pos_embed.data


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
    def __init__(self, config):
        super(Trtr, self).__init__()
        dp = 0 if config.task == "pretrain" else 0.1
        net_cfg = {"d_model": config.d_model,
                   "n_heads": config.n_heads,
                   "num_encoder_layers":config.e_layers,
                   "num_decoder_layers":config.d_layers,
                   "activation":config.activation,
                   "dropout":dp}
        if config.use_relative:
            self.batch_first = True
            Embedding = RltvEmbedding
            Net = Rltv
        else:
            self.batch_first = False
            Embedding = NrmlEmbedding
            Net = nn.Transformer
        self.pred_len = config.pred_len
        self.enc_embeding = Embedding(config)
        self.dec_embeding = Embedding(config)
        self.trtr = Net(**net_cfg)
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.d_reduction = nn.Conv1d(in_channels=config.d_model, out_channels=(config.max_car_num*4),
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.criterion = nn.MSELoss()

    def forward(self, enc_x, dec_x, gt_x):
        enc_token = self.enc_embeding(enc_x)
        dec_token = self.dec_embeding(dec_x)
        if self.batch_first:
            output = self.trtr(enc_token, dec_token).permute(0, 2, 1)
        else:
            output = self.trtr(enc_token.permute(1, 0, 2), dec_token.permute(1, 0, 2)).permute(1, 2, 0)
        outputs = self.d_reduction(output).permute(0, 2, 1)  # -> batch, seq_len, d_model
        outputs = outputs[:, -self.pred_len:, :]
        loss = self.criterion(outputs, gt_x)
        return outputs, loss