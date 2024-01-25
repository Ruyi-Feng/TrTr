
from pretrain.dataset import Dataset_flatten_simu, Dataset_stack_simu
from pretrain.exp import Exp_Main
import torch.distributed as dist
import torch



class Simulator(Exp_Main):

    dataset_factory = {
        "flatten": Dataset_flatten_simu,
        "stack": Dataset_stack_simu,
    }

    def __init__(self, args, local_rank=-1):
        args.batch_size = 1
        super(Simulator, self).__init__(args, local_rank)

    def pred(self, enc_x, dec_x, gt_x):
        enc_x = torch.tensor(enc_x)
        dec_x = torch.tensor(dec_x)
        gt_x = torch.tensor(gt_x)
        enc_x = enc_x.float().to(self.device)
        dec_x = dec_x.float().to(self.device)
        gt_x = gt_x.float().to(self.device)
        output, _ = self.model(enc_x, dec_x, gt_x)
        return output.detach().cpu()

    def renew(self, enc_x, dec_x, output):
        """
        enc_x: torch.tensor   size = [batch, (input_len-pred_len)/2, c_in]
        dec_x: batch, (input_len+pred_len)/2, c_in
        output: batch, pred_len, c_out
        """
        seq = self.simu_data.trans.merge(enc_x[0], dec_x[0], output[0])
        enc_x, dec_x, gt_x = self.simu_data.trans.extend(seq)
        enc_x = enc_x.unsqueeze(0)
        dec_x = dec_x.unsqueeze(0)
        gt_x = gt_x.unsqueeze(0)
        return enc_x, dec_x, gt_x

    def run(self, sample_num=1, simu_len=10):
        if dist.get_rank() == 0:
            print("simulate start ...")
            simu_rslt = dict()
            self.simu_data, simu_loader = self._get_data()
            self.model.eval()
            with torch.no_grad():
                for i, (enc_x, dec_x, gt_x, gt_ori) in enumerate(simu_loader):
                    if i >= sample_num:
                        break
                    simulate = []
                    for j in range(simu_len):
                        print("sample: %d, test step: %d"%(i, j))
                        output = self.pred(enc_x, dec_x, gt_x)
                        enc_x, dec_x, gt_x = self.renew(enc_x, dec_x, output)
                        simulate = simulate + output[0].tolist()
                    simu_rslt.update({i: {"groundtruth": gt_ori[0][int(self.args.input_len-self.args.pred_len):].tolist(), "simulate": simulate}})
            return simu_rslt

