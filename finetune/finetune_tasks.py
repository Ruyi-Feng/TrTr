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

    def _mask(self, tokens, mask_scheme) -> typing.List[list]:
        mask_scheme = dict(mask_scheme)
        token_under_mask = []
        current_span = 0
        for i in range(len(tokens)):
            if i in mask_scheme:
                current_span = mask_scheme[i]
            if current_span > 0:
                current_span -= 1
                token_under_mask.append(tokens[i, :].tolist())
                tokens[i, :] = self._mask_token
        return tokens, np.array(token_under_mask)

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

    def _pad(self, gt, x, pad_len):
        for i in range(pad_len):
            gt = np.concatenate((gt, x[self.input_len + i, :]), axis=0)
        return gt

    def _process(self, x):
        """
        return: enc, dec, gt
        enc/dec: 被按照泊松mask，真值放在gt里，不超过self.pred_len个
        gt: 存放mask的真值，末尾用未来值补上。
        """
        # [[frame, id, x, y, w, h], ...]
        # 泊松挖空，总共随机挖n个，n小于10，10-n补到末尾
        enc = x[:self.input_len]
        total_msk = random.randint(self.pred_len/2, self.pred_len)
        spans = self._gen_spans(total_msk)  # 生成总数不超过10个的span
        pad_len = self.pred_len - sum(spans)   # 计算需要补上多少个未来值
        n_spans = len(spans)
        n_possible_insert_poses = self.input_len - sum(spans) - n_spans + 1
        abs_insert_poses = sorted(random.sample(
            range(n_possible_insert_poses), n_spans))
        mask_scheme = self._distribute_insert_poses(abs_insert_poses, spans)
        mask_scheme = self._random_add_one(mask_scheme)
        enc, gt = self._mask(enc, mask_scheme)
        gt = self._pad(gt, x, pad_len)  # 给gt增加pad的未来值
        dec = enc.copy()
        return enc, dec, gt


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
