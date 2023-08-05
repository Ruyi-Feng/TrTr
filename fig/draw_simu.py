
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




def visual_frm(pd):
    scale = 40
    img = np.zeros((10*scale, 60*scale, 3), np.uint8)
    img.fill(255)
    for i in range(len(pd)//4):
        j = i * 4
        pd_x1, pd_y1, pd_x2, pd_y2 = pd[j+0], pd[j+1], pd[j+0]+pd[j+2], pd[j+1]+pd[j+3]
        print(pd_x1)
        print(pd_y1)
        cv2.rectangle(img, (int(pd_x1*scale), int(pd_y1*scale)), (int(pd_x2*scale), int(pd_y2*scale)), (0, 0, 255), 2)
    img = img[0:6*scale, 9*scale: 23*scale]
    cv2.resize(img, dsize=None, fx=2.5, fy=2.5)
    cv2.imshow('img', img)
    cv2.waitKey(0)

def visual_tra(x, y, ax1, ax2):
    if x[0] > x[-1]:
        x = abs(x * -1)
        # plt.subplot(1, 2, 1)
        # plt.plot(track, 'b')
        ax1.plot3D(np.arange(len(x)), y, x)

    else:
        # plt.subplot(1, 2, 2)
        # plt.plot(x, 'b')
        ax2.plot3D(np.arange(len(x)), y, x)

def draw_frms(simu):
    for line in simu:
        visual_frm(line)

def smooth(x):
    return np.convolve(x, np.ones(7), 'valid')

def draw_track(simu):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    for i in range(len(simu[0])//4):
        sx = smooth(simu[:, i*4])
        sy = smooth(simu[:, i*4 + 1])
        visual_tra(sx, sy, ax1, ax2)
    plt.show()

def draw_section(simu):
    fig = plt.figure()
    for i in range(len(simu[0])//4):
        sx = smooth(simu[:, i*4])
        sy = smooth(simu[:, i*4 + 1])
        if sy[0] > 17: #or sy[0] <10:
            continue
        plt.plot(sx)
    plt.show()


simu = np.load("./results/simulate-9-200.npy")
draw_track(simu)
# draw_frms(simu)

# draw_section(simu)

