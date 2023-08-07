from finetune.dataset_base import Dataset_Base
import numpy as np
import math
import random
import torch
import typing

MaskScheme = typing.List[typing.Tuple[int, int]]


class Data_Compensation(Dataset_Base):
    def __init__(self, index_path: str=".\\data\\index.bin",
                 data_path:str=".\\data\\data.bin",
                 max_car_num: int=40,
                 input_len: int=5,
                 pred_len: int=1):
        super(Data_Compensation, self).__init__(index_path, data_path, max_car_num, input_len, pred_len)
        self._poisson_rate = 3
        self._max_span_len = 5
        self._mask_token = np.ones(max_car_num*4) * -1
        self._poisson_dist = self._build_poisson_dist()

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

    def _pad_tail(self, x):
        gt_x = x[-self.pred_len:, :]
        enc_x = x[:self.pred_len, :].tolist()
        [enc_x.append(self._mask_token) for i in range(self.pred_len)]
        return np.array(enc_x), gt_x

    def _process(self, x):
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
        enc_x = np.array(enc_x)
        dec_x = enc_x.copy()
        return enc_x, dec_x, gt_x


class Data_Prediction(Dataset_Base):
    def __init__(self, index_path: str=".\\data\\index.bin",
                 data_path:str=".\\data\\data.bin",
                 max_car_num: int=40,
                 input_len: int=5,
                 pred_len: int=1):
        super(Data_Compensation, self).__init__(index_path, data_path, max_car_num, input_len, pred_len)
        self._mask_token = np.ones(max_car_num*4) * -1

    def _process(self, x):
        gt_x = x[-self.pred_len:, :]
        enc_x = x[:self.pred_len, :].tolist()
        [enc_x.append(self._mask_token) for i in range(self.pred_len)]
        enc_x = np.array(enc_x)
        return enc_x, enc_x.copy(), gt_x

class Data_Summary(Dataset_Base):
    def __init__(self, index_path: str=".\\data\\index.bin",
                 data_path:str=".\\data\\data.bin",
                 max_car_num: int=40,
                 input_len: int=5,
                 pred_len: int=1):
        super(Data_Summary, self).__init__(index_path, data_path, max_car_num, input_len, pred_len)

    def _process(self, x):
        # [[frame, id, x, y, w, h], ...]
        pass


class Data_Control(Dataset_Base):
    def __init__(self, index_path: str=".\\data\\index.bin",
                 data_path:str=".\\data\\data.bin",
                 max_car_num: int=40,
                 input_len: int=5,
                 pred_len: int=1):
        super(Data_Control, self).__init__(index_path, data_path, max_car_num, input_len, pred_len)

    def _process(self, x):
        # [[frame, id, x, y, w, h], ...]
        pass
