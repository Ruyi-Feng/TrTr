import copy
import math
import numpy as np
import random
import torch
import typing
from adapter.device import torch_npu

MaskScheme = typing.List[typing.Tuple[int, int]]


class PrcsBase():
    def __init__(self, noise_rate: float = 0.3,
                 msk_rate: float = 0.3,
                 poisson_rate: int = 3,
                 max_span_len: int = 5,
                 max_car_num: int = 10,
                 input_len: int = 20,
                 pred_len: int = 10) -> None:
        """
        noise_rate:
        ----------
        float = 0.5
        the input x is randomly select to add noise by noise_rate

        msk_rate:
        --------
        float = 0.5
        when add noise by randomly mask vehicles of one frame,
        msk_rate percent of vehicles will be masked in one frame.
        """
        self.noise_rate = noise_rate
        self.msk_rate = msk_rate
        self._poisson_rate = poisson_rate
        self._max_span_len = max_span_len
        self.input_len = input_len
        self.pred_len = pred_len
        self._poisson_dist = self._build_poisson_dist()
        self._mask_token = (np.ones(max_car_num*4) * -1).tolist()

    def _build_poisson_dist(self) -> torch.distributions.Categorical:
        lambda_to_the_k = 1
        e_to_the_minus_lambda = math.exp(-self._poisson_rate)
        k_factorial = 1
        ps = []
        for k in range(0, self._max_span_len + 1):
            ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
            lambda_to_the_k *= self._poisson_rate
            k_factorial *= k + 1
            if ps[-1] < 0.0000001:
                break
        ps = torch.FloatTensor(ps)
        return torch.distributions.Categorical(ps)

    def _pocess_len(self, seq_len: int, pocess_rate: float) -> int:
        x = seq_len * pocess_rate
        integer_part = int(x)
        fractional_part = x - float(integer_part)
        should_add = random.random() < fractional_part
        should_mask_len = integer_part + should_add
        return should_mask_len

    def _distribute_insert_poses(self, abs_insert_poses: typing.List[int], spans: typing.List[int]) -> MaskScheme:
        offset = 0
        mask_scheme = []
        for abs_insert_pos, span in zip(abs_insert_poses, spans):
            insert_pos = abs_insert_pos + offset
            mask_scheme.append((insert_pos, span))
            offset += span + 1
        return mask_scheme

    def _random_add_one(self, mask_scheme: MaskScheme) -> MaskScheme:
        should_add_one = random.random() < 0.5
        if should_add_one:
            mask_scheme = [(insert_pos + 1, span)
                           for insert_pos, span in mask_scheme]
        return mask_scheme

    def _gen_spans(self, should_mask_len: int) -> typing.List[int]:
        spans = self._poisson_dist.sample((should_mask_len,))
        spans_cum = torch.cumsum(spans, dim=0)
        idx = torch.searchsorted(spans_cum, should_mask_len).item()
        spans = spans[:idx].tolist()
        if idx > spans_cum.size(0) - 1:
            return spans
        if idx - 1 < 0:
            return [self._poisson_rate]
        last = should_mask_len - spans_cum[idx - 1].item()
        if last > 0:
            spans.append(last)
        return spans

    def _if_noise(self) -> bool:
        return random.random() < self.noise_rate


class Pad_tail(PrcsBase):
    def __init__(self, noise_rate: float = 0.3,
                 msk_rate: float = 0.3,
                 poisson_rate: int = 3,
                 max_span_len: int = 5,
                 max_car_num: int = 10,
                 input_len: int = 20,
                 pred_len: int = 10) -> None:
        """
        noise_rate:
        ----------
        float = 0.5
        the input x is randomly select to add noise by noise_rate

        msk_rate:
        --------
        float = 0.5
        when add noise by randomly mask vehicles of one frame,
        msk_rate percent of vehicles will be masked in one frame.
        """
        super(Pad_tail, self).__init__(noise_rate, msk_rate, poisson_rate,
                                   max_span_len, max_car_num, input_len, pred_len)

    def _mask(self, tokens: typing.List[list], mask_scheme: MaskScheme) -> typing.List[list]:
        mask_scheme = dict(mask_scheme)
        masked_tokens = []
        gt_tokens = []
        current_span = 0
        for i, t in enumerate(tokens):
            if i in mask_scheme:
                current_span = mask_scheme[i]
            if current_span > 0:
                current_span -= 1
                masked_tokens.append(self._mask_token)
                gt_tokens.append(t)
                continue
            masked_tokens.append(t)
        return masked_tokens, gt_tokens

    def _pad_tail(self, x):
        gt_x = x[-self.pred_len:, :]
        enc_x = x[:-self.pred_len, :].tolist()
        [enc_x.append(self._mask_token) for i in range(self.pred_len)]
        return np.array(enc_x), gt_x

    def _add_noise(self, x) -> list:
        """
        x: frame sequence
        - 选msk_rate的frm padding掉, padding内容成为gt
        - msk 长度不足的在末尾补同样长度的pad

        return
        ------
        enc_x, gt_x
        """
        seq_len = len(x)
        msk_len = self._pocess_len(seq_len, self.msk_rate)
        if msk_len < self._poisson_rate:
            return self._pad_tail(x)

        if msk_len < self.pred_len:
            tail_msk_len = self.pred_len - msk_len
            seq_len = seq_len - tail_msk_len
            tail_x = x[-tail_msk_len:, :]
            x = x[:-tail_msk_len, :]
        elif msk_len > self.pred_len:
            tail_msk_len = 0
            msk_len = self.pred_len

        spans = self._gen_spans(msk_len)
        n_spans = len(spans)
        n_possible_insert_poses = seq_len - sum(spans) - n_spans + 1
        sample_method = random.sample if n_possible_insert_poses > n_spans else random.choices
        abs_insert_poses = sorted(sample_method(
            range(n_possible_insert_poses), k=n_spans))
        mask_scheme = self._distribute_insert_poses(abs_insert_poses, spans)
        mask_scheme = self._random_add_one(mask_scheme)
        enc_x, gt_x = self._mask(x.tolist(), mask_scheme)
        gt_x = np.concatenate((gt_x, tail_x), axis=0)
        [enc_x.append(self._mask_token) for i in range(tail_msk_len)]
        return np.array(enc_x), gt_x

    def derve(self, x):
        """
        x: x, y, w, h

        return
        ------
        enc_x: np.array -> torch(size=[batch, input_len, c_in])
        gt_x : np.array -> torch(size=[batch, pred_len, c_in])
        """
        x_copy = copy.deepcopy(x)
        if self._if_noise():
            enc_x, gt_x = self._add_noise(x_copy)
        else:
            enc_x, gt_x = self._pad_tail(x_copy)
        dec_x = copy.deepcopy(enc_x)
        return enc_x, dec_x, gt_x


class Mask(PrcsBase):
    def __init__(self, noise_rate: float = 0.3,
                 msk_rate: float = 0.05,
                 poisson_rate: int = 3,
                 max_span_len: int = 5,
                 max_car_num: int = 10,
                 input_len: int = 20,
                 pred_len: int = 10) -> None:
        """
        noise_rate:
        ----------
        float = 0.5
        the input x is randomly select to add noise by noise_rate

        msk_rate:
        --------
        float = 0.5
        when add noise by randomly mask vehicles of one frame,
        msk_rate percent of vehicles will be masked in one frame.
        """
        super(Mask, self).__init__(noise_rate, msk_rate, poisson_rate,
                                   max_span_len, max_car_num, input_len, pred_len)

    def _mask(self, tokens: typing.List[list], mask_scheme: MaskScheme) -> typing.List[list]:
        mask_scheme = dict(mask_scheme)
        masked_tokens = []
        current_span = 0
        for i, t in enumerate(tokens):
            if i in mask_scheme:
                current_span = mask_scheme[i]
            if current_span > 0:
                current_span -= 1
                masked_tokens.append(self._mask_token)
                continue
            masked_tokens.append(t)
        return masked_tokens

    def _add_noise(self, x) -> list:
        """
        x: frame sequence
        - 选msk_rate的frm padding掉, padding内容成为gt
        - msk 长度不足的在末尾补同样长度的pad

        return
        ------
        enc_x
        """
        seq_len = len(x)
        msk_len = self._pocess_len(seq_len, self.msk_rate)
        if msk_len < self._poisson_rate:
            return x
        spans = self._gen_spans(msk_len)
        n_spans = len(spans)
        n_possible_insert_poses = seq_len - sum(spans) - n_spans + 1
        sample_method = random.sample if n_possible_insert_poses > n_spans else random.choices
        abs_insert_poses = sorted(sample_method(
            range(n_possible_insert_poses), k=n_spans))
        mask_scheme = self._distribute_insert_poses(abs_insert_poses, spans)
        mask_scheme = self._random_add_one(mask_scheme)
        enc_x = self._mask(x.tolist(), mask_scheme)
        return np.array(enc_x)

    def derve(self, x):
        """
        x: x, y, w, h

        return
        ------
        enc_x: np.array -> torch(size=[batch, input_len, c_in])
        gt_x : np.array -> torch(size=[batch, pred_len, c_in])
        """
        x_copy = copy.deepcopy(x)
        gt_x = copy.deepcopy(x)
        if self._if_noise():
            enc_x = self._add_noise(x_copy)
        else:
            enc_x = x_copy
        dec_x = copy.deepcopy(enc_x)
        return enc_x, dec_x, gt_x


class Predict(PrcsBase):
    def __init__(self, noise_rate: float = 0.3,
                 msk_rate: float = 0.3,
                 poisson_rate: int = 3,
                 max_span_len: int = 5,
                 max_car_num: int = 10,
                 input_len: int = 20,
                 pred_len: int = 10) -> None:
        """
        noise_rate:
        ----------
        float = 0.5
        the input x is randomly select to add noise by noise_rate

        msk_rate:
        --------
        float = 0.5
        when add noise by randomly mask vehicles of one frame,
        msk_rate percent of vehicles will be masked in one frame.
        """
        super(Predict, self).__init__(noise_rate, msk_rate, poisson_rate,
                                    max_span_len, max_car_num, input_len, pred_len)

    def _mask(self, tokens: typing.List[list], mask_scheme: MaskScheme) -> typing.List[list]:
        mask_scheme = dict(mask_scheme)
        masked_tokens = []
        current_span = 0
        for i, t in enumerate(tokens):
            if i in mask_scheme:
                current_span = mask_scheme[i]
            if current_span > 0:
                current_span -= 1
                masked_tokens.append(self._mask_token)
                continue
            masked_tokens.append(t)
        return np.array(masked_tokens)

    def _trans_type(self) -> int:
        return random.randint(0, 1)

    def _noise_one(self, x):
        seq_len = len(x)
        del_len = self._pocess_len(seq_len, self.msk_rate)
        if del_len < self._poisson_rate:
            return x.copy()
        spans = self._gen_spans(del_len)
        n_spans = len(spans)
        n_possible_insert_poses = seq_len - sum(spans) - n_spans + 1
        sample_method = random.sample if n_possible_insert_poses > n_spans else random.choices
        abs_insert_poses = sorted(sample_method(
            range(n_possible_insert_poses), k=n_spans))
        mask_scheme = self._distribute_insert_poses(abs_insert_poses, spans)
        mask_scheme = self._random_add_one(mask_scheme)
        return self._mask(x.tolist(), mask_scheme)

    def _noise_two(self, x) -> np.array:
        ids = random.sample(range(0, len(x)), 2)
        min_id, max_id = sorted(ids)
        pre_frms = x[:min_id]
        post_frms = x[max_id+1:]
        frm1 = x[min_id].reshape(1, -1)
        frm2 = x[max_id].reshape(1, -1)
        inter_frms = x[min_id + 1: max_id]
        return np.concatenate((pre_frms, frm2, inter_frms, frm1, post_frms), axis=0)

    def _add_noise(self, x) -> list:
        """
        x: frame sequence
        - _noise_1 选msk_rate的frm padding掉
        - _noise_2 交换两帧位置
        """
        self.frm_num = len(x)
        self.trans_type = self._trans_type()
        # print("trans_type", trans_type)
        if self.trans_type == 0:
            return self._noise_one(x)
        elif self.trans_type == 1:
            return self._noise_two(x)

    def derve(self, x):
        """
        x: input_len + pred_len
        -> x:[: input_len], gt:[input_len: input_len + pred_len]

        return
        ------
        enc_x: np.array -> torch(size=[batch, input_len, c_in]) with noise
        dec_x: np.array -> torch(size=[batch, input_len, c_in]) without noise
        gt_x : np.array -> torch(size=[batch, pred_len, c_in])
        """
        x, gt = x[: self.input_len - self.pred_len], x[self.input_len - self.pred_len: self.input_len]
        dec_x = copy.deepcopy(x)
        x_copy = copy.deepcopy(x)
        if self._if_noise():
            enc_x = self._add_noise(x_copy)
        else:
            enc_x = x_copy
        return enc_x, dec_x, gt


class Selfregression(PrcsBase):
    def __init__(self, noise_rate: float = 0.3,
                 msk_rate: float = 0.3,
                 poisson_rate: int = 3,
                 max_span_len: int = 5,
                 max_car_num: int = 10,
                 input_len: int = 20,
                 pred_len: int = 10) -> None:
        """
        noise_rate:
        ----------
        float = 0.5
        the input x is randomly select to add noise by noise_rate

        msk_rate:
        --------
        float = 0.5
        when add noise by randomly mask vehicles of one frame,
        msk_rate percent of vehicles will be masked in one frame.
        """
        super(Selfregression, self).__init__(noise_rate, msk_rate, poisson_rate,
                                    max_span_len, max_car_num, input_len, pred_len)

    def architecture(self):
        return "selfregression"

    def derve(self, x):
        """
        x: x, y, w, h
        decoder only stucturte
        with only decoder backbone
        0123 input
        1234 pred

        return
        ------
        enc_x: np.array -> torch(size=[batch, input_len-1, c_in])
        gt_x : np.array -> torch(size=[batch, pred_len-1, c_in])
        """
        gt_x = copy.deepcopy(x[1:])
        dec_x = copy.deepcopy(x[:-1])
        enc_x = copy.deepcopy(x[:-1])
        return enc_x, dec_x, gt_x


class Histregression(PrcsBase):
    def __init__(self, noise_rate: float = 0.3,
                 msk_rate: float = 0.3,
                 poisson_rate: int = 3,
                 max_span_len: int = 5,
                 max_car_num: int = 10,
                 input_len: int = 20,
                 pred_len: int = 10) -> None:
        super(Histregression, self).__init__(noise_rate, msk_rate, poisson_rate,
                                    max_span_len, max_car_num, input_len, pred_len)

    def architecture(self):
        return "histregression"

    def derve(self, x):
        """
        x: x, y, w, h
        using history trajectory regression
        hist input in encoder
        expected prediction input in decoder self-regression
        gt indicate dec_x seq + 1

        return
        ------
        enc_x: np.array -> torch(size=[batch, input_len-pred_len, c_in])
        gt_x : np.array -> torch(size=[batch, pred_len-1, c_in])
        """
        enc_end = self.input_len - self.pred_len
        dec_start = enc_end
        enc_x = copy.deepcopy(x[:enc_end])
        dec_x = copy.deepcopy(x[dec_start:-1])
        gt_x = copy.deepcopy(x[dec_start+1:])
        return enc_x, dec_x, gt_x


class Histseq2seq(PrcsBase):
    def __init__(self, noise_rate: float = 0.3,
                 msk_rate: float = 0.3,
                 poisson_rate: int = 3,
                 max_span_len: int = 5,
                 max_car_num: int = 10,
                 input_len: int = 20,
                 pred_len: int = 10) -> None:
        super(Histseq2seq, self).__init__(noise_rate, msk_rate, poisson_rate,
                                    max_span_len, max_car_num, input_len, pred_len)

    def architecture(self):
        return "histseq2seq"

    def derve(self, x):
        """
        x: x, y, w, h
        using history trajectory regression
        hist input in encoder
        decoder is zeros like vector
        gt indicate following sequence

        return
        ------
        enc_x: np.array -> torch(size=[batch, input_len-pred_len, c_in])
        gt_x : np.array -> torch(size=[batch, pred_len, c_in])
        """
        enc_end = self.input_len - self.pred_len
        dec_start = enc_end
        enc_x = copy.deepcopy(x[:enc_end])
        dec_x = np.zeros_like(x[dec_start:])
        gt_x = copy.deepcopy(x[dec_start:])
        return enc_x, dec_x, gt_x


class Perfix(PrcsBase):
    def __init__(self, noise_rate: float = 0.3,
                 msk_rate: float = 0.3,
                 poisson_rate: int = 3,
                 max_span_len: int = 5,
                 max_car_num: int = 10,
                 input_len: int = 20,
                 pred_len: int = 10) -> None:
        """
        noise_rate:
        ----------
        float = 0.5
        the input x is randomly select to add noise by noise_rate

        msk_rate:
        --------
        float = 0.5
        when add noise by randomly mask vehicles of one frame,
        msk_rate percent of vehicles will be masked in one frame.
        """
        super(Perfix, self).__init__(noise_rate, msk_rate, poisson_rate,
                                    max_span_len, max_car_num, input_len, pred_len)

    def architecture(self):
        return "perfix"

    def derve(self, x):
        """
        x: x, y, w, h
        decoder only stucturte
        with only decoder backbone
        0123 input
        1234 pred

        return
        ------
        dec_x: np.array -> torch(size=[batch, pred_len-1, c_in])
        gt_x : np.array -> torch(size=[batch, pred_len-1, c_in])
        """
        gt_x = copy.deepcopy(x[1:])
        dec_x = copy.deepcopy(x[:-1])
        enc_x = copy.deepcopy(x[:-1])
        return enc_x, dec_x, gt_x


class Histlabel(PrcsBase):
    def __init__(self, noise_rate: float = 0.3,
                 msk_rate: float = 0.3,
                 poisson_rate: int = 3,
                 max_span_len: int = 5,
                 max_car_num: int = 10,
                 input_len: int = 20,
                 pred_len: int = 10) -> None:
        super(Histlabel, self).__init__(noise_rate, msk_rate, poisson_rate,
                                    max_span_len, max_car_num, input_len, pred_len)

    def architecture(self):
        return "histlabel"

    def derve(self, x):
        """
        x: x, y, w, h
        using history trajectory regression
        hist input in encoder
        half of the decodwer is as zeros like vector and half is labeled
        gt indicate zeros like sequence

        return
        ------
        enc_x: np.array -> torch(size=[batch, (input_len-pred_len)/2, c_in])
        dec_x: np.array -> torch(size=[batch, pred_len, c_in])
        gt_x : np.array -> torch(size=[batch, pred_len, c_in])
        """
        enc_end = int((self.input_len - self.pred_len) / 2)
        dec_start = enc_end
        label_end = self.input_len - self.pred_len
        enc_x = copy.deepcopy(x[:enc_end])
        dec_x = copy.deepcopy(x[dec_start:])
        dec_x[-self.pred_len:] = 0
        gt_x = copy.deepcopy(x[-self.pred_len:])
        return enc_x, dec_x, gt_x

    def merge(self, enc, dec, gt):
        """
        enc: [batch, (input_len-pred_len)/2, c_in]
        dec: [(input_len+pred_len)/2, c_in]
        gt : [pred_len, c_in]
        with batch dim
        """
        seq = torch.cat((enc, dec[:self.pred_len], gt), 0)
        return seq

    def extend(self, seq):
        """
        seq -> new enc_x, dec_x
        """
        valid_len = self.input_len - self.pred_len
        gt_x = torch.zeros(self.pred_len, seq.shape[1])
        extended_seq = seq[-valid_len:]
        enc_end = int((self.input_len - self.pred_len) / 2)
        enc_x = copy.deepcopy(extended_seq[:enc_end])
        dec_x = torch.cat((extended_seq[enc_end:], gt_x), 0)
        return enc_x, dec_x, gt_x


class Mae(PrcsBase):
    def __init__(self, noise_rate: float = 0.3,
                 msk_rate: float = 0.75,
                 poisson_rate: int = 3,
                 max_span_len: int = 5,
                 max_car_num: int = 10,
                 input_len: int = 20,
                 pred_len: int = 10) -> None:
        super(Mae, self).__init__(noise_rate, msk_rate, poisson_rate,
                                    max_span_len, max_car_num, input_len, pred_len)

    def architecture(self):
        return "mae"

    def derve(self, x):
        """
        x: x, y, w, h
        enc-dec but without cross-attention

        return
        ------
        all gt x
        mask in network
        """
        return x, x, copy.deepcopy(x)


class Env_gpt_seq2seq(PrcsBase):
    def __init__(self, noise_rate: float = 0.3,
                 msk_rate: float = 0.3,
                 poisson_rate: int = 3,
                 max_span_len: int = 5,
                 max_car_num: int = 10,
                 input_len: int = 20,
                 pred_len: int = 10) -> None:
        super(Env_gpt_seq2seq, self).__init__(noise_rate, msk_rate, poisson_rate,
                                    max_span_len, max_car_num, input_len, pred_len)

    def architecture(self):
        return "env_gpt"

    def derve(self, x):
        """
        x: x, y, w, h
        encoder only with gpt

        return
        ------
        enc_x: np.array -> torch(size=[batch, input_len, c_in])
        dec_x: np.array -> torch(size=[batch, input_len, c_in])
        gt_x : np.array -> torch(size=[batch, pred_len, c_in])
        """
        enc_x = torch.zeros(self.input_len, x.shape[1])
        label_end = self.input_len - self.pred_len
        dec_x = copy.deepcopy(x[:label_end])
        gt_x = copy.deepcopy(x[label_end:])
        return enc_x, dec_x, gt_x

class Probe(PrcsBase):
    def __init__(self, noise_rate: float = 0.3,
                 msk_rate: float = 0.3,
                 poisson_rate: int = 3,
                 max_span_len: int = 5,
                 max_car_num: int = 10,
                 input_len: int = 20,
                 pred_len: int = 10) -> None:
        super(Probe, self).__init__(noise_rate, msk_rate, poisson_rate,
                                    max_span_len, max_car_num, input_len, pred_len)
        """
        此处msk rate当作probe rate用了, msk_rate为实际保留下来的探针比例
        非探针部分的pred_len部分全部填0
        """

    def architecture(self):
        return "probe"

    def _choose_probe(self, seq_len):
        probe_list = []
        for i in range(seq_len // 4):
            if random.random() >= self.msk_rate:
                probe_list.append(i)
        return probe_list

    def _generate_probe(self, non_probe_list, x, label_end):
        for i in non_probe_list:
            x[label_end:, i: i+4] = 0
        return x

    def _probing(self, x, label_end):
        non_probe_list = self._choose_probe(x.shape[1])
        return self._generate_probe(non_probe_list, x, label_end)

    def derve(self, x):
        """
        x: x, y, w, h
        encoder only with gpt

        return
        ------
        enc_x: np.array -> torch(size=[batch, input_len, c_in])
        dec_x: np.array -> torch(size=[batch, input_len, c_in])
        gt_x : np.array -> torch(size=[batch, pred_len, c_in])
        """
        enc_x = torch.zeros(self.input_len, x.shape[1])
        label_end = self.input_len - self.pred_len
        dec_x = self._probing(copy.deepcopy(x), label_end)
        gt_x = copy.deepcopy(x[label_end:])
        return enc_x, dec_x, gt_x
