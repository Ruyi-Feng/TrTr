from finetune.exp import Exp_Ft
from finetune.params import params
import os


if __name__ == '__main__':
    args = params()
    args.task = "compensation"
    print("---------init finetune main---------------")
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    local_rank = int(os.environ['LOCAL_RANK'])
    trtr = Exp_Ft(args, local_rank)
    trtr.train()
