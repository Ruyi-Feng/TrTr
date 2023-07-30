from trtr.net import Trtr
import os
import numpy as np
from datetime import timedelta
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def load_model(args, dir, local_rank):
    dist.init_process_group(backend='nccl', timeout=timedelta(days=1))
    device = torch.device('cuda', local_rank)
    model = Trtr(args).float().to(device)
    if os.path.exists(dir + 'checkpoint_best.pth'):
        print("load checkpoints best", dir)
        model.load_state_dict(torch.load(dir + 'checkpoint_best.pth', map_location=torch.device('cpu')))
    return device, DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

def pred(model, device, enc_x, dec_x, gt_x):
    enc_x = torch.tensor(enc_x)
    dec_x = torch.tensor(dec_x)
    gt_x = torch.tensor(gt_x)
    enc_x = enc_x.float().to(device)
    dec_x = dec_x.float().to(device)
    output, loss = model(enc_x, dec_x, gt_x)
    return output.detach().cpu()


def load(flnm) -> dict:
    with open(flnm, 'r') as load_f:
        info = json.load(load_f)
    return info


def init_data(rslt_path, sample=1):
    info = load(rslt_path)
    i = 0
    for k, v in info.items():
        if i < sample:
            i += 1
            continue
        enc = v["enc"][0]
        gt = v["gt"][0]
        break
    enc = np.expand_dims(np.array(enc), axis=0)
    return enc, enc.copy(), gt

def renew(enc_x, output):
    new_enc = np.concatenate((enc_x[0][10:, :], output[0]), axis=0)
    new_enc = np.expand_dims(new_enc, axis=0)
    new_dec = new_enc.copy()
    return new_enc, new_dec

def run(args, rslt_path, sample, local_rank=-1, loops=30):
    print("save path", args.save_path)
    device, model = load_model(args, args.save_path+"pretrain/", local_rank)
    enc_x, dec_x, gt_x = init_data(rslt_path, sample)
    print("enc shape", enc_x.shape)
    simulate = []
    for step in range(loops):
        print("----%d-----"%step)
        output = pred(model, device, enc_x, dec_x, gt_x)
        print("output shape", output.shape)
        # print("enc_x last 10", enc_x[0])
        # print("output ", output[0])
        enc_x, dec_x = renew(enc_x, output)
        simulate = simulate + output[0].tolist()
    return simulate


