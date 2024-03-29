from datetime import timedelta
import numpy as np
import os
from pretrain.dataset import Dataset_flatten, Dataset_stack
import time
import torch
from torch import optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from trtr.net import Trtr, MAE
from adapter.device import torch_npu
from adapter.device import amp
from adapter.device import copy_from_local
from adapter.device import device_label
from adapter.device import backend_label
from adapter.npu2gpu import save as cpu_save
import copy


class Exp_Main:

    dataset_factory = {
        "flatten": Dataset_flatten,
        "stack": Dataset_stack,
    }

    def __init__(self, args, local_rank=-1):
        self.args = args
        if self.args.task != "pretrain":
            self.args.save_path = self.args.save_path + 'finetune/' + self.args.task + '/'
        else:
            self.args.save_path = self.args.save_path + 'pretrain/'
        self.best_score = None
        self.WARMUP = args.warmup_steps
        self.device = torch.device(device_label, local_rank)
        self.local_rank = local_rank
        if torch_npu is not None:
            torch_npu.npu.set_device(local_rank)
        else:
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend_label, timeout=timedelta(days=1))  # init_method = "tcp://127.0.0.1:21210", rank=local_rank, world_size=8, 
        if args.sepecific is not None:
            self.model = self._task_model()
        else:
            self.model = self._build_model()

    def _task_model(self):
        if self.args.architecture == 'mae':
            model = MAE(self.args).float().to(self.device)
        else:
            model = Trtr(self.args).float().to(self.device)
        if os.path.exists(self.args.sepecific):
            model.load_state_dict(torch.load(self.args.sepecific, map_location=torch.device('cpu')))
        return DDP(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

    def _build_model(self):
        if self.args.architecture == 'mae':
            model = MAE(self.args).float().to(self.device)
        else:
            model = Trtr(self.args).float().to(self.device)
        if os.path.exists(self.args.save_path + 'checkpoint_best.pth'):
            model.load_state_dict(torch.load(self.args.save_path + 'checkpoint_best.pth', map_location=torch.device('cpu')))
        elif os.path.exists(self.args.save_path + 'checkpoint_last.pth'):
            model.load_state_dict(torch.load(self.args.save_path + 'checkpoint_last.pth', map_location=torch.device('cpu')))
        return DDP(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

    def _get_data(self):
        batch_sz = self.args.batch_size
        Dataset_pretrain = self.dataset_factory[self.args.data_form]
        data_set = Dataset_pretrain(index_path=self.args.index_path,
                                    data_path=self.args.data_path,
                                    max_car_num=self.args.max_car_num,
                                    input_len=self.args.input_len,
                                    pred_len=self.args.pred_len,
                                    architecture=self.args.architecture,
                                    frm_embed=self.args.frm_embed,
                                    id_embed=self.args.id_embed)
        sampler = None
        drop_last = False
        if self.args.is_train:
            sampler = DistributedSampler(data_set)
            drop_last=self.args.drop_last
            # shuffle = True
        data_loader = DataLoader(data_set, batch_size=batch_sz, sampler=sampler, drop_last=drop_last, pin_memory=True)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        lamda1 = lambda step: 1 / np.sqrt(max(step , self.WARMUP))
        scheduler = optim.lr_scheduler.LambdaLR(model_optim, lr_lambda=lamda1, last_epoch=-1)
        return model_optim, scheduler

    def _save_model(self, vali_loss, path):
        if self.best_score is None:
            self.best_score = vali_loss
            torch.save(self.model.module.state_dict(), path)
        elif vali_loss < self.best_score:
            self.best_score = vali_loss
            torch.save(self.model.module.state_dict(), path)

    def vali(self, vali_data, vali_loader):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (enc_x, dec_x, gt_x) in enumerate(vali_loader):
                enc_x = enc_x.float().to(self.device)
                dec_x = dec_x.float().to(self.device)
                gt_x = gt_x.float().to(self.device)
                if amp is not None:
                    with amp.autocast():
                        outputs, loss = self.model(enc_x, dec_x, gt_x)
                else:
                    outputs, loss = self.model(enc_x, dec_x, gt_x)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        _, train_loader = self._get_data()
        # vali_data, vali_loader = self._get_data()

        time_now = time.time()
        train_steps = len(train_loader)
        path = self.args.save_path + 'checkpoint_'

        model_optim, scheduler = self._select_optimizer()
        if amp is not None:
            scaler = amp.GradScaler()
        else:
            scaler = None

        for epoch in range(self.args.train_epochs):
            train_loader.sampler.set_epoch(epoch)
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (enc_x, dec_x, gt_x) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                enc_x = enc_x.float().to(self.device)
                dec_x = dec_x.float().to(self.device)
                gt_x = gt_x.float().to(self.device)

                if amp is not None:
                    with amp.autocast():
                        _, loss = self.model(enc_x, dec_x, gt_x)
                else:
                    _, loss = self.model(enc_x, dec_x, gt_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                scheduler.step()

            if dist.get_rank() == 0:
                train_loss = np.average(train_loss)
                # vali_loss = self.vali(vali_data, vali_loader)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, 0.00))

                # saving model
                self._save_model(train_loss, path + 'best.pth')
                if torch_npu is not None:
                    save_model = copy.deepcopy(self.model.module)
                    cpu_save(save_model, path + 'cpu_best.pth')
            dist.barrier()
        if dist.get_rank() == 0:
            torch.save(self.model.module.state_dict(), path + 'last.pth')
            print("=============success save last checkpoints============")
            if torch_npu is not None:
                cpu_save(self.model.module, path + 'cpu_last.pth')
        # print("rank_number", dist.get_rank())
        dist.barrier()
        dist.destroy_process_group()

    def test(self):
        if dist.get_rank() == 0:
            test_data, test_loader = self._get_data()
            self.model.eval()
            result = dict()
            with torch.no_grad():
                for i, (enc_x, dec_x, gt_x) in enumerate(test_loader):
                    print("test step: %d"%i)
                    enc_x = enc_x.float().to(self.device)
                    dec_x = dec_x.float().to(self.device)
                    gt_x = gt_x.float().to(self.device)
                    output, loss = self.model(enc_x, dec_x, gt_x)
                    output = output.detach().cpu().tolist()
                    gt_x = gt_x.detach().cpu().tolist()
                    enc_x = enc_x.detach().cpu().tolist()  # batch, seq_len, d_model
                    dec_x = dec_x.detach().cpu().tolist()
                    result.update({i: {"gt": gt_x, "pd": output, "enc": enc_x, "dec": dec_x}})
                    if i > 50:
                        break
            return result