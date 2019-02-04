from django.conf import settings
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pickle
from .algorithm_conf import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import time
import numpy as np
import os
from os import path
from skimage import feature, exposure
import cv2
import json
from datetime import datetime, timedelta
import gc
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def str_to_tuple(a_str):
    return tuple((int(a_str.split(',')[0]), int(a_str.split(',')[1])))


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


def execute_hog_pic(pic_size, orientations, pixels_per_cell, cells_per_block, is_color, user_id):
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


# TODO: judege whether to recalculate
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

    # X_train, X_test, y_train, y_test = list(X_train), list(X_test), list(y_train), list(y_test)

    # initial feature_vector.pki
    feature_vector = {
        user_id: {
            'X_train': X_train,
            'X_test': X_test,
            'Y_train': y_train,
            'Y_test': y_test
        }
    }

    with open(settings.FEATURE_VECTOR_PATH, "wb") as f:
        pickle.dump(feature_vector, f)

    # return X_train, X_test, y_train, y_test
    return None


def execute_contrast_algorithm(user_id, is_standard, contrast_algorithm):
    num_folds = 10
    seed = 7
    scoring = 'accuracy'

    with open(settings.FEATURE_VECTOR_PATH, "rb") as load_f:
        feature_vector = pickle.load(load_f)

    models = {}
    if is_standard:
        models = {}
        models['ScalerSVM'] = Pipeline([('Scaler', StandardScaler()), ('SVM', SVC(gamma='scale'))])
        models['ScalerLR'] = Pipeline([('Scaler', StandardScaler()), ('LR', LogisticRegression(solver='lbfgs'))])
        models['ScalerKNN'] = Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])
        models['ScalerCART'] = Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeClassifier())])
        models['ScalerNB'] = Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])
    else:
        models['SVM'] = SVC(gamma='scale')
        models['LR'] = LogisticRegression(solver='lbfgs')
        models['KNN'] = KNeighborsClassifier()
        models['CART'] = DecisionTreeClassifier()
        models['NB'] = GaussianNB()

    results = []
    for key in models:
        kfold = KFold(n_splits=num_folds, random_state=seed)

        # TODO:The nums of samples should big than kfold
        cv_results = cross_val_score(models[key],
                                     feature_vector[user_id]['X_train'],
                                     feature_vector[user_id]['Y_train'],
                                     cv=kfold,
                                     scoring=scoring)
        results.append(cv_results)

    # 评估算法 - 箱线图
    fig = plt.figure(0)
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(models.keys())

    contrast_algorithm_path = path.join(settings.MEDIA_ROOT, 'algorithm', 'hog_picture',
                                        user_id + '_contrast_algorithm.png')
    plt.savefig(contrast_algorithm_path)
    # plt.close(0)

    return None
#
#
# def test_get_pic_vector(user_id):
#     BASE_DIR = '/Users/sweeney/WorkSpaces/Git/svm_cat_web'
#     MEDIA_ROOT = BASE_DIR + '/media'
#     ALGORITHM_JSON_PATH = MEDIA_ROOT + '/algorithm' + '/algorithm_info.json'
#
#     # load the algorithm_info.json
#     with open(ALGORITHM_JSON_PATH, "r") as load_f:
#         algorithm_info = json.load(load_f)
#
#     pic_root_dir = os.path.join(MEDIA_ROOT, 'upload_images', str(user_id))
#
#     positive_category = algorithm_info[user_id]['model_para']['category_positive']
#     negative_category = algorithm_info[user_id]['model_para']['category_negative']
#     pic_size = algorithm_info[user_id]['pic_para']['pic_size']
#     orientations = int(algorithm_info[user_id]['pic_para']['orientations'])
#     pixels_per_cell = algorithm_info[user_id]['pic_para']['pixels_per_cell']
#     cells_per_block = algorithm_info[user_id]['pic_para']['cells_per_block']
#     is_color = algorithm_info[user_id]['pic_para']['is_color']
#     validation_size = algorithm_info[user_id]['model_para']['validation_size']
#
#     pic_size = str_to_tuple(pic_size)
#     pixels_per_cell = str_to_tuple(pixels_per_cell)
#     cells_per_block = str_to_tuple(cells_per_block)
#
#     img_list = []
#     positive_pic = [os.path.join(pic_root_dir, positive_category, i) for i in
#                     os.listdir(os.path.join(pic_root_dir, positive_category))]
#     negative_pic = [os.path.join(pic_root_dir, negative_category, i) for i in
#                     os.listdir(os.path.join(pic_root_dir, negative_category))]
#     pic_file_list = positive_pic + negative_pic
#
#     for pic in pic_file_list:
#         pic = os.path.join(pic_root_dir, pic)
#         a_pic = cv2.imread(pic, is_color)
#         a_pic = cv2.resize(a_pic, pic_size, interpolation=cv2.INTER_AREA)
#         img_list.append(a_pic)
#
#     pic_vector = hog(img_list, orientations, pixels_per_cell, cells_per_block)
#     label = np.array(np.repeat(1, pic_vector.shape[0]))
#     label[len(positive_pic):] = 0
#
#     X_train, X_test, y_train, y_test = train_test_split(pic_vector, label, test_size=validation_size, random_state=7)
#
#     # X_train, X_test, y_train, y_test = list(X_train), list(X_test), list(y_train), list(y_test)
#
#     feature_vector = {}
#
#     feature_vector[user_id] = {
#         'X_train': X_train,
#         'X_test': X_test,
#         'y_train': y_train,
#         'y_test': y_test
#     }
#
#     pickle.dump(feature_vector, open('feature_vector.pkl', 'wb'))
#
#     # return X_train, X_test, y_train, y_test
#     return None
#
#
# test_get_pic_vector('1')
#
# data = pickle.load(open('feature_vector.pkl', 'rb'))
# print(data['1']['y_test'])
