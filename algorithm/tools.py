import cv2
import time
import numpy as np


def cal_hog_time(img, orientations, pixels_per_cell, cells_per_block):

    # opencv
    size = pixels_per_cell[0] * cells_per_block[0]
    win_size = (size, size)
    block_size = pixels_per_cell
    block_stride = pixels_per_cell
    cell_size = pixels_per_cell
    nbins = orientations

    first_time = time.clock()

    # for i in range(1):
    #     hog_vector = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    #     hist_2 = hog_vector.compute(img)
    #     hist_2 = np.array(hist_2).reshape(len(hist_2), )

    hog_vector = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hist_2 = hog_vector.compute(img)
    hist_2 = np.array(hist_2).reshape(len(hist_2), )

    print(id(hog_vector))

    end_time = time.clock()

    avg_time = (end_time - first_time)/1
    return avg_time

