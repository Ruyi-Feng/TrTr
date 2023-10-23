import numpy as np
from pretrain.trans import Mask
from pretrain.trans import Histseq2seq, Histregression, Selfregression, Predict, Pad_tail, Mask
from torch.utils.data import Dataset
from adapter.device import torch_npu


data_factory = {
    "histseq2seq": Histseq2seq,
    "histregression": Histregression,
    "selfregression": Selfregression,
    "predict": Predict,
    "mask": Mask,
    "pad_tail": Pad_tail,
}

class Dataset_Pretrain(Dataset):
    def __init__(self, index_path: str=".\\data\\index.bin",
                 data_path:str=".\\data\\data.bin",
                 max_car_num: int=40,
                 input_len: int=5,
                 pred_len: int=1,
                 architecture: str="histseq2seq"):
        Data_formate = data_factory[architecture]
        self.trans = Data_formate(max_car_num=max_car_num, input_len=input_len, pred_len=pred_len)  # max_seq_len=max_seq_len
        self.LONG_SCALE = 10
        self.LATI_SCALE = 10
        self.SIZE_SCALE = 10
        self.max_car_num = max_car_num  # c_in = max_car_num * 4
        self.input_len = input_len
        self.pred_len = pred_len
        self.idx_path = index_path
        self.data_path = data_path
        self.train_idx = []
        self.dataset_length = 0
        self.f_data = open(self.data_path, 'rb')
        for line in open(self.idx_path, 'rb'):
            line = line.decode().split()[0].split(',')
            self.train_idx.append([line[0], int(line[1]), int(line[2])])
            self.dataset_length += 1


    def _trans_to_array(self, info: str) -> np.array:
        """
        把数据组织好，除了挖空的
        组织两个dict
        1. {carid: list([[], []])}
        2. {frame: set()}
        return x, gt
        """
        trj_per_car, car_per_frm = self._load(info)
        continue_car = self._intersection(car_per_frm)
        x = self._form_dataset(continue_car, trj_per_car)
        return self._pre_process(x)

    def _intersection(self, car_dict: dict, intersection=None) -> set:
        for k in car_dict:
            if intersection is None:
                intersection = car_dict[k]
            else:
                intersection = intersection.intersection(car_dict[k])
        new_inter = set()
        for k in intersection:
            new_inter.add(k)
            if len(new_inter) >= self.max_car_num:
                break
        return new_inter

    def __getitem__(self, index: int):
        """
        return:
        enc_x, dec_x, gt_x
        """
        head, tail = self.train_idx[index][1], self.train_idx[index][2]
        self.f_data.seek(head)
        info = self.f_data.read(tail - head)
        seq = self._trans_to_array(info)  # xywh xywh ... (standared)
        enc, dec, gt = self.trans.derve(seq)  # for mask of compensation
        return enc, dec, gt

    def __len__(self):
        return self.dataset_length


class Data_flatten(Dataset_Pretrain):
    def __init__(self, index_path: str=".\\data\\index.bin",
                 data_path:str=".\\data\\data.bin",
                 max_car_num: int=40,
                 input_len: int=5,
                 pred_len: int=1,
                 architecture: str="histseq2seq"):
        super().__init__(index_path, data_path, max_car_num, input_len, pred_len, architecture)

    def _load(self, info: str) -> tuple:
        trj_per_car = {}
        car_per_frm = {}
        lines = info.split()
        for line in lines:
            line = line.decode().split(',')
            line_data = []   # frm, car_id, xywh
            for i, item in enumerate(line):
                item = float(item)
                if i == 0:
                    car_per_frm.setdefault(item, set())  # {frame: set()}
                if i == 1:
                    car_per_frm[float(line[0])].add(item)
                line_data.append(item)
            trj_per_car.setdefault(line_data[1], list())  # {car_id: list()}
            trj_per_car[line_data[1]].append(line_data)
        return trj_per_car, car_per_frm

    def _form_dataset(self, continue_car: set, trj_per_car: dict) -> np.array:
        # 原本是把每辆车的轨迹横着粘贴起来
        total_data = []
        pad_num = self.max_car_num - len(continue_car)
        # print("pad_num", pad_num, "continues", len(continue_car))
        tmp = np.zeros((self.input_len, 4))
        for car in continue_car:
            tmp = np.array(trj_per_car[car])[:, 2:]
            if len(total_data):
                total_data = np.concatenate((total_data, tmp), axis=1)
            else:
                total_data = tmp
        for pad in range(pad_num):
            if len(total_data) == 0:
                total_data = tmp
                continue
            total_data = np.concatenate((total_data, np.zeros(tmp.shape)), axis=1)
        return total_data

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
        return sec[:self.input_len]


class Dataset_stack(Dataset_Pretrain):
    def __init__(self, index_path: str=".\\data\\index.bin",
                 data_path:str=".\\data\\data.bin",
                 max_car_num: int=10,
                 input_len: int=5,
                 pred_len: int=1,
                 architecture: str="histseq2seq"):
        super().__init__(index_path, data_path, max_car_num, input_len, pred_len, architecture)


    def _load(self, info: str) -> typing.Tuple[list, dict]:
        data_list = []
        car_dict = {}
        lines = info.split()
        for line in lines:
            line = line.decode().split(',')
            line_data = []
            for i in range(len(line)):
                item = float(line[i])
                if i == 0:
                    car_dict.setdefault(item, set())  # {frame: set()}
                if i == 1:
                    car_dict[float(line[0])].add(item)
                line_data.append(item)
            data_list.append(line_data)
        return data_list, car_dict


    def _form_dataset(self, car_set: set, data_list: list) -> np.array:
        new_data = []
        for line in data_list:
            if line[1] in car_set:
                new_data.append(line)
        return np.array(new_data)

    def _pre_process(self, sec):
        """
        sec: np.array
        [
            [frm, car_id, xywh],
            [frm, car_id, xywh],
        ]
        """
        if len(sec) < 1:
            return sec
        sec[:, 2] = sec[:, 2] / self.LONG_SCALE
        sec[:, 3] = sec[:, 3] / self.LATI_SCALE
        sec[:, 4] = sec[:, 4] / self.SIZE_SCALE
        sec[:, 5] = sec[:, 5] / self.SIZE_SCALE
        # 这里input_len怎么设置需要注意！！！！
        return sec[:self.input_len]
