from finetune.dataset_base import Dataset_Base
import random


class Data_Compensation(Dataset_Base):
    def __init__(self, index_path: str=".\\data\\index.bin",
                 data_path:str=".\\data\\data.bin",
                 max_car_num: int=40,
                 input_len: int=5,
                 pred_len: int=1):
        super(Data_Compensation, self).__init__(index_path, data_path, max_car_num, input_len, pred_len)

    def _process(self, x):
        # [[frame, id, x, y, w, h], ...]
        pass


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
