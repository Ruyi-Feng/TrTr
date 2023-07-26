import copy
import math
import numpy as np
import random
import torch
import typing

MaskScheme = typing.List[typing.Tuple[int, int]]


class Mask():
    def __init__(self, noise_rate: float = 0.3,
                 msk_rate: float = 0.1,
                 poisson_rate: int = 3,
                 max_span_len: int = 5,
                 max_car_num: int = 10) -> None:
        """
        noise_rate:
        ----------
        float = 0.5
        the input x is randomly select to add noise by noise_rate

        del_rate:
        --------
        float = 0.5
        when add noise by randomly delete vehicles of one frame,
        del_rate percent of vehicles will be deleted in one frame.

        msk_rate:
        --------
        float = 0.5
        when add noise by randomly mask vehicles of one frame,
        msk_rate percent of vehicles will be masked in one frame.
        """
        # self.MAX_CAR_NUM = 80
        self.LONG_SCALE = 300
        self.LATI_SCALE = 100
        self.SIZE_SCALE = 20
        self.noise_rate = noise_rate
        self.msk_rate = msk_rate
        self._poisson_rate = poisson_rate
        self._max_span_len = max_span_len
        self._poisson_dist = self._build_poisson_dist()
        self._mask_token = np.ones(max_car_num*4).tolist()

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

    def _random_add_one(self, mask_scheme: MaskScheme) -> MaskScheme:
        should_add_one = random.random() < 0.5
        if should_add_one:
            mask_scheme = [(insert_pos + 1, span)
                           for insert_pos, span in mask_scheme]
        return mask_scheme

    def _if_noise(self) -> bool:
        return random.random() < self.noise_rate

    def _pre_process(self, sec):
        """
        sec: np.array
        [
            [xywh1, xywh2, ...],
            [xywh1, xywh2, ...],
        ]
        """
        if len(sec) < 1:
            return sec
        loop = len(sec[0]) // 4
        for i in range(loop):
            sec[:, i * 4 + 0] = sec[:, i * 4 + 0] / self.LONG_SCALE
            sec[:, i * 4 + 1] = sec[:, i * 4 + 1] / self.LATI_SCALE
            sec[:, i * 4 + 2] = sec[:, i * 4 + 2] / self.SIZE_SCALE
            sec[:, i * 4 + 3] = sec[:, i * 4 + 3] / self.SIZE_SCALE
        return sec


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
        abs_insert_poses = sorted(random.sample(
            range(n_possible_insert_poses), n_spans))
        mask_scheme = self._distribute_insert_poses(abs_insert_poses, spans)
        mask_scheme = self._random_add_one(mask_scheme)
        return self._mask(x.tolist(), mask_scheme)

    def _noise_two(self, x) -> np.array:
        cur_frame = -1
        ids = random.sample(range(0, len(x)), 2)
        min_id, max_id = sorted(ids)
        pre_frms = x[:min_id]
        post_frms = x[max_id+1:]
        frm1 = x[min_id]
        frm2 = x[max_id]
        inter_frms = x[min_id + 1: max_id]
        return np.concatenate((pre_frms, frm2, inter_frms, frm1, post_frms), axis=0)

    def _add_noise(self, x) -> list:
        """
        x: frame sequence
        - _noise_0 选msk_rate的frm padding掉
        - _noise_1 交换两帧位置
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
        x: x, y, w, h

        return
        ------
        enc_x: np.array -> torch(size=[batch, seq_len, c_in])
        dec_x: np.array -> torch(size=[batch, 5])

        enc_mark means how many lines in each frames in enc_x
        """
        x = self._pre_process(x)
        dec_x = copy.deepcopy(x)
        x_copy = copy.deepcopy(x)
        if self._if_noise():
            enc_x = self._add_noise(x_copy)
        else:
            enc_x = x_copy
        return enc_x, dec_x
