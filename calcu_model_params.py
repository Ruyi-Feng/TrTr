import torch
import os
from trtr.net import Trtr
from pretrain.params import params


def load(local_rank, pth):
    args = params()
    device = torch.device('cuda', local_rank)
    model = Trtr(args).float().to(device)
    if os.path.exists(pth):
        model.load_state_dict(torch.load(pth, map_location=torch.device('cpu')))
    return model

def calcu_param_num(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    pth = './checkpoints/cpu_histreg_d512_in120_pd60_loss0.092_epoch100.pth'
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    local_rank = int(os.environ['LOCAL_RANK'])
    model = load(local_rank, pth)
    print(calcu_param_num(model))

