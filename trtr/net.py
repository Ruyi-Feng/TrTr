import torch.nn as nn
import torch
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
        net_cfg = {"d_model": config.d_model,
                   "n_head": config.n_heads,
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
        dp = 0 if config.task == "pretrain" else 0.1
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

class CompTrtr(Trtr):
    def __init__(self, config):
        """
        训练数据格式和TrTr完全保持一致
        enc/dec: batch * seq_len * d_model = batch * 20 * 40
        gt: batch * seq_len * d_model = batch * 10 * 40
        任务设计 enc/dec 中用mask表示挖空，不足10个的补在后面做预测
        gt为挖空的值，总共十个，不足以预测补。
        """
        return super(CompTrtr, self).__init__(config)

class SumTrtr(Trtr):
    def __init__(self, config):
        super(SumTrtr, self).__init__(config)
        self.summaryer = self._make_layers(config)

    def _make_layers(self, config):
        # 此处是否增加几层卷积存疑
        # d_model * input_len -> d_model -> d_model -> 3
        layers = []
        layers += [nn.Linear(config.d_model * config.input_len, config.d_model),
                   nn.ReLU(inplace=True)]
        layers += [nn.Linear(config.d_model, config.d_model),
                   nn.ReLU(inplace=True)]
        layers += [nn.Linear(config.d_model, 3)]
        return nn.Sequential(*layers)

    def forward(self, enc_x, dec_x, gt_x):
        """
        enc/dec: batch * seq_len * d_model = batch * 20 * 40
        gt: batch * 1 * 3: [K, Q, V]
        """
        enc_token = self.enc_embeding(enc_x)
        dec_token = self.dec_embeding(dec_x)
        features = self.trtr(enc_token.permute(1, 0, 2), dec_token.permute(1, 0, 2)).permute(1, 2, 0)  # batch, d_model, seq_len
        outputs = self.summaryer(torch.flatten(features, start_dim=1))
        loss = self.criterion(outputs, gt_x)
        return outputs, loss

class CtrlTrtr(Trtr):
    def __init__(self, config):
        super(CtrlTrtr, self).__init__(config)
        self.controler = self._make_layers(config)
        self.signal_embeding = nn.Embedding(config.max_signal_num, config.d_model)
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.d_reduction_control = nn.Conv1d(in_channels=config.d_model, out_channels=config.d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.criterion_control = nn.MSELoss()

    def _make_layers(self, config):
        # d_model * input_len -> d_model -> d_model -> 3
        # 这里预计加一个encoder，用于生成控制后的序列
        pass

    def forward(self, enc_x, dec_x, signal, gt_x):
        """
        这个怎么练存疑，因为原始模型已经能学习到车辆未来会运行的状态了，要考虑好信号加入对模型的影响。
        enc/dec: batch * seq_len * d_model = batch * 20 * 40
        signal: batch * 1 * car_num   沿着seq_len都一致，每个car一个signal，表示它在gt的部分是加速还是减速信号。
        gt: batch * 10 * 40
        """
        enc_token = self.enc_embeding(enc_x)
        dec_token = self.dec_embeding(dec_x)
        features = self.trtr(enc_token.permute(1, 0, 2), dec_token.permute(1, 0, 2))
        features = features + self.signal_embeding(signal)
        outputs = self.controler(features).permute(1, 2, 0)  # seq_len, batch, d_model
        outputs = self.d_reduction_control(outputs).permute(0, 2, 1)  # batch, seq_len, d_model
        loss = self.criterion_control(outputs, gt_x)
        return outputs, loss
