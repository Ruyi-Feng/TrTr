import numpy as np
from pretrain.mask import Mask
from torch.utils.data import Dataset


class Dataset_Pretrain(Dataset):
    def __init__(self, index_path: str=".\\data\\index.bin",
                 data_path:str=".\\data\\data.bin",
                 max_car_num: int=40,
                 input_len: int=5,
                 pred_len: int=1):
        self.trans = Mask(max_car_num=max_car_num)  # max_seq_len=max_seq_len
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

    def _intersection(self, car_dict: dict, intersection=None) -> set:
        for k in car_dict:
            if intersection is None:
                intersection = car_dict[k]
            else:
                intersection = intersection.intersection(car_dict[k])
        new_inter = set()
        for k in intersection:
            new_inter.add(k)
            if len(new_inter) >= self.max_car_num - 1:
                break
        return new_inter

    def _form_dataset(self, continue_car: set, trj_per_car: dict) -> np.array:
        total_data = np.array()
        pad_num = self.max_car_num - len(continue_car)
        for car in continue_car:
            tmp = np.array(trj_per_car[car])[:, 2:]
            if len(total_data):
                total_data = np.concatenate((total_data, tmp), axis=1)
            else:
                total_data = tmp
        for pad in range(pad_num):
            total_data = np.concatenate((total_data, np.zeros(tmp.shape)), axis=1)
        return total_data

    def _trans_to_array(self, info: str) -> list:
        """
        把数据组织好，除了挖空的
        组织两个dict
        1. {carid: set()}
        2. {frame: set()}
        return x, gt
        """
        trj_per_car, car_per_frm = self._load(info)
        continue_car = self._intersection(car_per_frm)
        x = self._form_dataset(continue_car, trj_per_car)
        return x[: self.input_len], x[self.input_len: self.input_len + self.pred_len]

    def __getitem__(self, index: int):
        """
        return:
        enc_x, dec_x, gt_x
        """
        head, tail = self.train_idx[index][1], self.train_idx[index][2]
        self.f_data.seek(head)
        info = self.f_data.read(tail - head)
        x, gt = self._trans_to_array(info)  # frame, id, x, y, w, h
        enc, dec = self.trans.derve(x)  # 这个是用来随机挖空的
        return enc, dec, gt

    def __len__(self):
        return self.dataset_length
