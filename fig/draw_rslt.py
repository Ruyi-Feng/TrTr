import json
import cv2
import numpy as np


def load(flnm) -> dict:
    with open(flnm, 'r') as load_f:
        info = json.load(load_f)
    return info

def visual(gt, pd):
    scale = 40
    img = np.zeros((10*scale, 60*scale, 3), np.uint8)
    for i in range(len(gt)//4):
        j = i * 4
        gt_x1, gt_y1, gt_x2, gt_y2 = gt[j+0], gt[j+1], gt[j+0]+gt[j+2], gt[j+1]+gt[j+3]
        print((gt_x1, gt_y1, gt_x2, gt_y2))
        pd_x1, pd_y1, pd_x2, pd_y2 = pd[j+0], pd[j+1], pd[j+0]+pd[j+2], pd[j+1]+pd[j+3]
        cv2.rectangle(img, (int(gt_x1*scale), int(gt_y1*scale)), (int(gt_x2*scale), int(gt_y2*scale)), (0, 255, 0), 1)
        cv2.rectangle(img, (int(pd_x1*scale), int(pd_y1*scale)), (int(pd_x2*scale), int(pd_y2*scale)), (0, 0, 255), 1)
    cv2.resize(img, dsize=None, fx=2.5, fy=2.5)
    cv2.imshow('img', img)
    cv2.waitKey(0)

def draw_frms(info):
    for k, v in info.items():
        for batch in range(len(v["gt"])):
            for i in range(len(v["gt"][batch])):
                visual(v["gt"][batch][i], v["pd"][batch][i])


info = load("./results/rslt.json")
draw_frms(info)
