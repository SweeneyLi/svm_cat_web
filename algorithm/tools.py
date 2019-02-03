from django.conf import settings

from .algorithm_conf import *

from sklearn.model_selection import train_test_split
import time
import numpy as np
import os
from os import path
from skimage import feature, exposure
import cv2
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta


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


def execute_hog_pic(pic_size, orientations, pixels_per_cell,cells_per_block, is_color, user_id):

    is_color_num = 1 if is_color else 0

    # 取出json中的test_pic, 放入图片参数
    with open(settings.ALGORITHM_JSON_PATH, "r") as load_f:
        algorithm_info = json.load(load_f)

    test_pic = algorithm_info[user_id]['pic_para']['test_pic']

    algorithm_info[user_id]['pic_para'] = {
        'test_pic': test_pic,
        'pic_size': pic_size,
        'orientations': orientations,
        'pixels_per_cell': pixels_per_cell,
        'cells_per_block': cells_per_block,
        'is_color': is_color_num
    }
    now = (datetime.now() + timedelta(hours=8)).strftime("%Y-%m-%d_%H:%M:%S.")
    algorithm_info[user_id]['update_time'] = now

    with open(settings.ALGORITHM_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(algorithm_info, f)

    # format the parameter
    pic_size = str_to_tuple(pic_size)
    pixels_per_cell = str_to_tuple(pixels_per_cell)
    cells_per_block = str_to_tuple(cells_per_block)

    # read pic and resize it
    saved_pic_path = path.join(settings.MEDIA_ROOT, 'algorithm', 'hog_picture',
                               user_id + '_' + 'hog_test_pic.jpg')

    img = cv2.imread(saved_pic_path, is_color_num)
    img = cv2.resize(img, pic_size, interpolation=cv2.INTER_AREA)
    # cv2.imwrite(saved_pic_path, img)

    # 将opencv的BGR模式转为matplotlib的RGB模式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # use the hog to change the picture
    fd, hog_image = feature.hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                cells_per_block=cells_per_block, multichannel=is_color,
                                block_norm='L2-Hys', visualize=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    # show the origin picture
    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)

    ax1.set_title('Input image: %s' % test_pic)

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # show hoged picture
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')

    # save the plt as png
    hog_picture_path = path.join(settings.MEDIA_ROOT, 'algorithm', 'hog_picture',
                                 user_id + '_hog_picture.png')
    plt.savefig(hog_picture_path)



def get_pic_vector(user_id):
    # load the algorithm_info.json
    with open(settings.ALGORITHM_JSON_PATH, "r") as load_f:
        algorithm_info = json.load(load_f)

    pic_root_dir = os.path.join(settings.MEDIA_ROOT, 'upload_images', str(user_id))
    positive_category = algorithm_info[user_id]['model_para']['category_positive']
    negative_category = algorithm_info[user_id]['model_para']['category_negative']
    pic_size = algorithm_info[user_id]['pic_para']['pic_size']
    orientations = int(algorithm_info[user_id]['pic_para']['orientations'])
    pixels_per_cell = algorithm_info[user_id]['pic_para']['pixels_per_cell']
    cells_per_block = algorithm_info[user_id]['pic_para']['cells_per_block']
    is_color = algorithm_info[user_id]['pic_para']['is_color']
    validation_size = algorithm_info[user_id]['model_para']['validation_size']

    pic_size = str_to_tuple(pic_size)
    pixels_per_cell = str_to_tuple(pixels_per_cell)
    cells_per_block = str_to_tuple(cells_per_block)

    img_list = []
    positive_pic = [os.path.join(pic_root_dir, positive_category, i) for i in
                    os.listdir(os.path.join(pic_root_dir, positive_category))]
    negative_pic = [os.path.join(pic_root_dir, negative_category, i) for i in
                    os.listdir(os.path.join(pic_root_dir, negative_category))]
    pic_file_list = positive_pic + negative_pic

    for pic in pic_file_list:
        pic = os.path.join(pic_root_dir, pic)
        a_pic = cv2.imread(pic, is_color)
        a_pic = cv2.resize(a_pic, pic_size, interpolation=cv2.INTER_AREA)
        img_list.append(a_pic)

    pic_vector = hog(img_list, orientations, pixels_per_cell, cells_per_block)
    label = np.array(np.repeat(1, pic_vector.shape[0]))
    label[len(positive_pic):] = 0

    X_train, X_test, y_train, y_test = train_test_split(pic_vector, label, test_size=validation_size, random_state=seed)
    return X_train, X_test, y_train, y_test
