from pretrain.exp import Exp_Main
from pretrain.params import params
import os
import json


if __name__ == '__main__':
    # --data_path ./data/data.bin
    # --index_path ./data/index.bin
    rslt_path = "./results/histseq2seq_local_epoch48.json"
    args = params()
    args.batch_size = 1
    args.is_train = False
    args.data_path = "./data/train/data_val_interval6_len30.bin"
    args.index_path = "./data/train/index_val_interval6_len30.bin"
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    local_rank = int(os.environ['LOCAL_RANK'])
    model = Exp_Main(args, local_rank)
    rslt = model.test()
    with open(rslt_path, "w") as f:
        rslt = json.dump(rslt, f)
