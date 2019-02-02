from django.conf import settings

import cv2
import time
import numpy as np
import os
import json


def str_to_tuple(a_str):
    return tuple((int(a_str.split(',')[0]), int(a_str.split(',')[1])))


def cal_hog_time(img, orientations, pixels_per_cell, cells_per_block):
    # opencv
    size = pixels_per_cell[0] * cells_per_block[0]
    win_size = (size, size)
    block_size = pixels_per_cell
    block_stride = pixels_per_cell
    cell_size = pixels_per_cell
    nbins = orientations

    first_time = time.clock()

    for i in range(10):
        hog_vector = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        hist_2 = hog_vector.compute(img)
        hist_2 = np.array(hist_2).reshape(len(hist_2), )
        del hist_2

    # hog_vector = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    # hist_2 = hog_vector.compute(img)
    # hist_2 = np.array(hist_2).reshape(len(hist_2), )

    end_time = time.clock()

    avg_time = (end_time - first_time)
    return avg_time


def hog(img_list, orientations, pixels_per_cell, cells_per_block):
    size = pixels_per_cell[0] * cells_per_block[0]
    win_size = (size, size)
    block_size = pixels_per_cell
    block_stride = pixels_per_cell
    cell_size = pixels_per_cell
    nbins = orientations
    hog_descriptor = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hist_list = []
    for a_img in img_list:
        hist = hog_descriptor.compute(a_img)
        hist = np.array(hist).reshape(len(hist), )
        hist_list.append(hist)
    return np.asarray(hist_list)


def get_pic_vector(user_id):
    # load the algorithm_info.json
    with open(settings.ALGORITHM_JSON_PATH, "r") as load_f:
        algorithm_info = json.load(load_f)

    pic_root_dir = os.path.join(settings.MEDIA_ROOT, 'upload_images', str(user_id))
    positive_category = algorithm_info[user_id]['test_category']['positive']
    negative_category = algorithm_info[user_id]['test_category']['negative']
    pic_size = algorithm_info[user_id]['pic_para']['pic_size']
    orientations = int(algorithm_info[user_id]['pic_para']['orientations'])
    pixels_per_cell = algorithm_info[user_id]['pic_para']['pixels_per_cell']
    cells_per_block = algorithm_info[user_id]['pic_para']['cells_per_block']
    # is_color = algorithm_info[user_id]['pic_para']['is_color']
    is_color = True

    pic_size = str_to_tuple(pic_size)
    pixels_per_cell = str_to_tuple(pixels_per_cell)
    cells_per_block = str_to_tuple(cells_per_block)

    img_list = []
    positive_pic = [os.path.join(pic_root_dir, positive_category, i) for i in
                    os.listdir(os.path.join(pic_root_dir, positive_category))]
    negative_pic = [os.path.join(pic_root_dir, negative_category, i) for i in
                    os.listdir(os.path.join(pic_root_dir, negative_category))]
    pic_file_list = positive_pic + negative_pic
    pic_color = 1 if is_color else 0
    for pic in pic_file_list:
        pic = os.path.join(pic_root_dir, pic)
        a_pic = cv2.imread(pic, pic_color)
        a_pic = cv2.resize(a_pic, pic_size, interpolation=cv2.INTER_AREA)
        img_list.append(a_pic)

    pic_vector = hog(img_list, orientations, pixels_per_cell, cells_per_block)
    label = np.array(np.repeat(1, pic_vector.shape[0]))
    label[len(positive_pic):] = 0

    return pic_vector, label
