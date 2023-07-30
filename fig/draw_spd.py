import matplotlib.pyplot as plt
import numpy as np
import json



def load(flnm) -> dict:
    with open(flnm, 'r') as load_f:
        info = json.load(load_f)
    return info

def diff(v, window):
    new_v = []
    for i in range(len(v)-window):
        new_v.append(v[i+window] - v[i])
    return new_v

def get_spd(gt, pd, window=3):
    return diff(gt, window), diff(pd, window)

def draw_spd(info):
    for k, v in info.items():
        max_car_num = len(info[k]["gt"][0][0]) // 4
        for batch in range(len(v["gt"])):
            for i in range(max_car_num):
                gt, pd = np.array(v["gt"][batch])[:, i * 4], np.array(v["pd"][batch])[:, i * 4]
                # v_gt, v_pd = get_spd(np.array(v["gt"][batch])[:, i * 4], np.array(v["pd"][batch])[:, i * 4])
                plt.figure()
                plt.plot(gt, c='g')
                plt.plot(pd, c='r')
                plt.show()




info = load("./results/rslt.json")
draw_spd(info)