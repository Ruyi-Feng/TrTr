from finetune.exp import Exp_Ft
from finetune.params import params as finetune_param
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json


class Evaluation:
    def __init__(self, if_pretrain: bool = False, task: str = "compensation"):
        self.args = finetune_param()
        self.args.data_path = './data/val/data.bin'
        self.args.index_path = './data/val/index.bin'
        self.args.batch_size = 2000
        self.args.is_train = False
        self.args.task = task
        if if_pretrain:
            self.args.save_path = './checkpoints/pretrain/'
        else:
            self.args.save_path = './checkpoints/finetune/'

    def obtain_results(self, rslt_path, local_rank):
        print("---------obtain finetune results---------------")
        bartti = Exp_Ft(self.args, local_rank)
        rslt = bartti.test()
        self._save(rslt, rslt_path)
        print("finish save results", self.args.task)

    def load(self, flnm) -> dict:
        with open(flnm, 'r') as load_f:
            info = json.load(load_f)
        return info

    def _save(self, rslt, rslt_path):
        """
        rslt: dict
        ----
        {
            i: {"gt": [], "pd": []},
        }
        gt: list
        [split(5 frm)
            [frm, car_id, x, y, w, h],
            [frm, car_id, x, y, w, h],
        ]
        """
        rslt = self._filt(rslt)
        with open(rslt_path, "w") as f:
            rslt = json.dump(rslt, f)

    def check_results(self, check_list: list = [], rslt_path: str = "./results/rslt.json"):
        if "rmse" in check_list:
            rmse = self._check_rmse(rslt_path)
            print("------%s-------" % self.args.task)
            print("pd gt rmse:", rmse)
        if "overlap" in check_list:
            num, total = self._check_overlap(rslt_path)
            print("------%s-------" % self.args.task)
            print("overlap num | total: ", num, total, num/total)
        if "spd_stable" in check_list:
            self._check_spd_stable(rslt_path)
        if "iou" in check_list:
            print("------%s-------" % self.args.task)
            print("average iou ", self._check_iou(rslt_path))
        if "DCN" in check_list:
            print("------%s-------" % self.args.task)
            print("DCN num: ", self._check_DCN(rslt_path))

    def _stand(self, gt, pd, enc):
        gt_new, pd_new, enc_new = [], [], []
        for i in range(len(gt)):
            if round(gt[i][0]) == 0:
                break
            gi = [round(gt[i][0]), round(gt[i][1])] + gt[i][2:]
            pi = [round(pd[i][0]), round(pd[i][1])] + pd[i][2:]
            enc_i = [round(enc[i][0]), round(enc[i][1])] + enc[i][2:]
            gt_new.append(gi)
            pd_new.append(pi)
            enc_new.append(enc_i)
        return gt_new, pd_new, enc_new

    def _filt(self, rslt):
        # rslt: batch, seq_len, d_model
        new_rslt = dict()
        for k, values in rslt.items():
            for batch in range(len(values["gt"])):
                gt, pd, enc = self._stand(
                    values["gt"][batch], values["pd"][batch], values["enc"][batch])
                new_rslt.update(
                    {str(k) + str(batch): {"gt": gt, "pd": pd, "enc": enc}})
        return new_rslt

    def _dis(self, a1, a2, w1, w2):
        return (abs(a1 - a2) - 0.5 * (w1 + w2)) < 0

    def _intersect(self, box1, box2):
        return (self._dis(box1[0], box2[0], box1[2], box2[2]) and self._dis(box1[1], box2[1], box1[3], box2[3]))

    def _overlaps(self, boxes):
        overlap = 0
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                if boxes[i][0] == 0:
                    continue
                overlap += self._intersect(boxes[i], boxes[j])
        return overlap

    def _count_overlap(self, pd):
        # 需要排除预测0值
        last_frm = 0
        overlap = 0
        boxes = []
        for line in pd:   # xywh xywh
            boxes = np.array(line).reshape((-1, 4))
            overlap += self._overlaps(boxes)
        return overlap

    def _check_overlap(self, save_path):
        info = self.load(save_path)
        """
        info: dict
        {k: {"gt": [], "pd": [], "enc": []}}
        """
        overlap = 0
        total_num = 0
        for k in info:
            for batch in range(len(info[k]["gt"])):
                total_num += (len(info[k]["gt"][batch][0])//4)
                overlap += self._count_overlap(info[k]["pd"][batch])
        return overlap, total_num

    def _check_rmse(self, rslt_path):
        # 返回真实m为单位的rmse
        info = self.load(rslt_path)
        """
        info: dict
        {k: {"gt": [], "pd": [], "enc": []}}
        """
        square = 0
        n = 0
        for k in info:
            for batch in range(len(info[k]["gt"])):
                square += np.sum(
                    np.square(np.array(info[k]["gt"][batch]) - np.array(info[k]["pd"][batch])))
                n += np.array(info[k]["gt"][batch]).size
        return np.sqrt(square / n) * 10

    def _draw_dv_distribution(self, dv):
        plt.figure()
        plt.grid()
        # sns.kdeplot(dv)
        percent_95 = np.percentile(dv, 95)
        print(percent_95)
        percent_5 = np.percentile(dv, 5)
        print(percent_5)
        sns.kdeplot(dv, color='r')
        plt.plot([percent_95, percent_95], [0, 1], 'b')
        plt.plot([percent_5, percent_5], [0, 1], 'b')
        # plt.hist(dv)
        plt.title("speed difference distribution <m>")
        plt.show()

    def _check_spd_stable(self, rslt_path):
        info = self.load(rslt_path)
        dv = []
        for k in info:
            max_car_num = len(info[k]["gt"][0][0]) // 4
            for batch in range(len(info[k]["gt"])):
                for i in range(max_car_num):
                    v_gt = abs(info[k]["gt"][batch][-1]
                               [i * 4] - info[k]["gt"][batch][0][i * 4])
                    v_pd = abs(info[k]["pd"][batch][-1]
                               [i * 4] - info[k]["pd"][batch][0][i * 4])
                    dv.append((v_gt - v_pd) * 3 * 10)
        self._draw_dv_distribution(dv)
        return dv

    def _iou(self, gt_box, b_box):
        width0 = gt_box[2]
        height0 = gt_box[3]
        width1 = b_box[2]
        height1 = b_box[3]
        max_x = max(gt_box[0]+0.5*width0, b_box[0]+0.5*width1)
        min_x = min(gt_box[0]-0.5*width0, b_box[0]-0.5*width1)
        width = width0 + width1 - (max_x - min_x)
        max_y = max(gt_box[1]+0.5*height0, b_box[1]+0.5*height1)
        min_y = min(gt_box[1]-0.5*height0, b_box[1]-0.5*height1)
        height = height0 + height1 - (max_y - min_y)
        if width <= 0 or height <= 0:
            return 0
        interArea = width * height
        boxAArea = width0 * height0
        boxBArea = width1 * height1
        if boxAArea + boxBArea - interArea == 0:
            return 1
        iou = interArea / (boxAArea + boxBArea - interArea)
        return iou

    def _calcu_iou(self, gt, pd):
        max_car_num = len(gt) // 4
        iou = 0
        num = 0
        for i in range(max_car_num):
            if gt[i*4] == 0:
                continue
            iou += self._iou(gt[i*4: i*4+4], pd[i*4: i*4+4])
            num += 1
        return iou, num

    def _check_iou(self, rslt_path):
        info = self.load(rslt_path)
        sum_iou = 0
        num = 0
        for k in info:
            for batch in range(len(info[k]["gt"])):
                for i in range(len(info[k]["gt"][batch])):
                    frm_gt = info[k]["gt"][batch][i]
                    frm_pd = info[k]["pd"][batch][i]
                    iou, n = self._calcu_iou(frm_gt, frm_pd)
                    sum_iou += iou
                    num += n
        print("sum_iou | total num", sum_iou, num)
        return sum_iou / num


    def _check_DCN(self, rslt_path):
        info = self.load(rslt_path)
        dcn_num = 0
        for k in info:
            max_car_num = len(info[k]["gt"][0][0]) // 4
            for batch in range(len(info[k]["gt"])):
                for i in range(max_car_num):
                    gt_f = (info[k]["gt"][batch][0][i*4] > 0)
                    pd_f = (info[k]["pd"][batch][0][i*4] > 0)
                    if gt_f != pd_f:
                        dcn_num += 1
        return dcn_num
