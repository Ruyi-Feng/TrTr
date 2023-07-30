from evaluation.simulation import run
from pretrain.params import params
import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    args = params()
    args.is_train = False
    print("---------init finetune main---------------")
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    local_rank = int(os.environ['LOCAL_RANK'])
    rslt_path = "./results/rslt.json"
    simulate = run(args, rslt_path, 3, local_rank)
    simulate = np.array(simulate)
    np.save("./results/simulate-3-300.npy", simulate)
