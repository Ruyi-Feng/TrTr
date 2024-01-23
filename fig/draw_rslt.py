import json
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load(flnm) -> dict:
    with open(flnm, 'r') as load_f:
        info = json.load(load_f)
    return info

def visual(gt, pd):
    scale = 40
    img = np.zeros((10*scale, 60*scale, 3), np.uint8)
    img.fill(255)
    for i in range(len(gt)//4):
        j = i * 4
        gt_x1, gt_y1, gt_x2, gt_y2 = gt[j+0], gt[j+1], gt[j+0]+gt[j+2], gt[j+1]+gt[j+3]
        print((gt_x1, gt_y1, gt_x2, gt_y2))
        pd_x1, pd_y1, pd_x2, pd_y2 = pd[j+0], pd[j+1], pd[j+0]+pd[j+2], pd[j+1]+pd[j+3]
        cv2.rectangle(img, (int(gt_x1*scale), int(gt_y1*scale)), (int(gt_x2*scale), int(gt_y2*scale)), (0, 255, 0), 2)
        cv2.rectangle(img, (int(pd_x1*scale), int(pd_y1*scale)), (int(pd_x2*scale), int(pd_y2*scale)), (0, 0, 255), 2)
    cv2.resize(img, dsize=None, fx=2.5, fy=2.5)
    cv2.imshow('img', img)
    cv2.waitKey(0)

def draw_frms(info):
    for k, v in info.items():
        for batch in range(len(v["gt"])):
            for i in range(len(v["gt"][batch])):
                visual(v["gt"][batch][i], v["pd"][batch][i])

def scatter(points, img, scale, c):
    car_num = len(points) // 4
    for i in range(car_num):
        px = points[i*4]
        py = points[i*4 +1]
        cv2.circle(img,(int(px*scale), int(py*scale)), 1, c, -1)
    return img

def draw_tracks(info):
    # draw enc
    scale = 80
    img = np.zeros((10*scale, 200*scale, 3), np.uint8)
    img.fill(255)
    for k, v in info.items():
        for batch in range(len(v["enc"])):
            for i in range(len(v["enc"][batch])):
                img = scatter(v["enc"][batch][i], img, scale, (255, 0, 0))

    for k, v in info.items():
        for batch in range(len(v["gt"])):
            for i in range(len(v["gt"][batch])):
                img = scatter(v["gt"][batch][i], img, scale, (0, 255, 0))
                img = scatter(v["pd"][batch][i], img, scale, (0, 0, 255))
    img = img[:, :40*scale]
    cv2.imshow('img', img)
    cv2.waitKey(0)

def draw_time_space(seq_pd, seq_gt, seq_enc, seq_dec):
    car_num = seq_pd.shape[1] // 4
    for i in range(car_num):
        plt.figure()
        print("len of seq_pd", len(seq_pd))
        print("len of seq_gt", len(seq_gt))
        print("len of seq_enc", len(seq_enc))
        plt.plot(np.arange(len(seq_enc), len(seq_enc)+len(seq_gt)), seq_gt[:, i * 4], c='g', label="gt")
        plt.plot(np.arange(len(seq_enc), len(seq_enc)+len(seq_pd)), seq_pd[:, i * 4], c='r', label="pd")
        plt.plot(np.arange(0, len(seq_enc)), seq_enc[:, i * 4], c='b')
        plt.plot(np.arange(len(seq_enc), len(seq_enc)+len(seq_dec)), seq_dec[:, i * 4], c='y', label="dec")
        plt.legend()
        plt.show()

def draw_traj(info):
    # single predicted trajectory
    for k, v in info.items():
        for batch in range(len(v["pd"])):
            seq_pd = np.array(v["pd"][batch])
            seq_gt = np.array(v["gt"][batch])
            seq_enc = np.array(v["enc"][batch])
            seq_dec = np.array(v["dec"][batch])
            draw_time_space(seq_pd, seq_gt, seq_enc, seq_dec)

info = load("./results/histlabel_local_epoch100.json")
draw_frms(info)
