from django.conf import settings
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from .models import SVMModel
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

import pickle
from .algorithm_conf import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import time
import numpy as np
import os
from os import path
from skimage import feature, exposure
import cv2
import json
import gc
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def hog(img_list, orientations, pixels_per_cell, cells_per_block):
    """
    change the img to feature_vector
    :param img_list:
    :param orientations:
    :param pixels_per_cell:
    :param cells_per_block:
    :return: vector(array)
    """

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
    """
    calculate the time of hog
    :param img:
    :param orientations:
    :param pixels_per_cell:
    :param cells_per_block:
    :return: time
    """

    # opencv
    size = pixels_per_cell[0] * cells_per_block[0]
    win_size = (size, size)
    block_size = pixels_per_cell
    block_stride = pixels_per_cell
    cell_size = pixels_per_cell
    nbins = orientations

    first_time = time.clock()

    for i in range(1000):
        hog_vector = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        hist_2 = hog_vector.compute(img)
        hist_2 = np.array(hist_2).reshape(len(hist_2), )
        del hist_2

    # hog_vector = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    # hist_2 = hog_vector.compute(img)
    # hist_2 = np.array(hist_2).reshape(len(hist_2), )

    end_time = time.clock()

    avg_time = (end_time - first_time) / 1000
    return avg_time


def execute_hog_pic(pic_size, orientations, pixels_per_cell, cells_per_block, is_color, user_id):
    """
    get the pic_para, save them to algorithm_info.json, then hog them by plot (save in local)
    :param pic_size:
    :param orientations:
    :param pixels_per_cell:
    :param cells_per_block:
    :param is_color:
    :param user_id:
    :return: None
    """

    # 取出json中的test_pic, 放入图片参数
    with open(settings.ALGORITHM_JSON_PATH, "r") as load_f:
        algorithm_info = json.load(load_f)

    test_pic = algorithm_info[user_id]['pic_para']['test_pic']

    # read pic and resize it
    saved_pic = saved_pic_path(user_id)
    if is_color:
        img = cv2.imread(saved_pic, 1)
        # 将opencv的BGR模式转为matplotlib的RGB模式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(saved_pic, 0)

    img = cv2.resize(img, pic_size, interpolation=cv2.INTER_AREA)

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
    hog_picture_path = hog_pic_path(user_id, relative=False)
    plt.savefig(hog_picture_path)
    plt.close('all')
    gc.collect()

    # hog_time = cal_hog_time(img, orientations, pixels_per_cell, cells_per_block)

    algorithm_info[user_id]['pic_para'].update({
        'test_pic': test_pic,
        'pic_size': str(pic_size),
        'orientations': orientations,
        'pixels_per_cell': str(pixels_per_cell),
        'cells_per_block': str(cells_per_block),
        'is_color': is_color
    })

    with open(settings.ALGORITHM_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(algorithm_info, f)
    gc.collect()

    return None


def get_pic_vector(user_id):
    """
    get the data_para and pic_para, change them to vector,
    then save in the feature_vector.pkl

    :param user_id: str
    :return: None
    """

    # TODO: judege whether to recalculate

    # load the algorithm_info.json
    with open(settings.ALGORITHM_JSON_PATH, "r") as load_f:
        algorithm_info = json.load(load_f)

    pic_root_dir = os.path.join(settings.MEDIA_ROOT, 'upload_images', str(user_id))

    positive_category = algorithm_info[user_id]['data_para']['category_positive']
    negative_category = algorithm_info[user_id]['data_para']['category_negative']
    pic_size = algorithm_info[user_id]['pic_para']['pic_size']
    orientations = int(algorithm_info[user_id]['pic_para']['orientations'])
    pixels_per_cell = algorithm_info[user_id]['pic_para']['pixels_per_cell']
    cells_per_block = algorithm_info[user_id]['pic_para']['cells_per_block']
    is_color = algorithm_info[user_id]['pic_para']['is_color']
    validation_size = algorithm_info[user_id]['data_para']['validation_size']

    pic_size = eval(pic_size)
    pixels_per_cell = eval(pixels_per_cell)
    cells_per_block = eval(cells_per_block)

    img_list = []
    positive_pic = [os.path.join(pic_root_dir, positive_category, i) for i in
                    os.listdir(os.path.join(pic_root_dir, positive_category))]
    negative_pic = [os.path.join(pic_root_dir, negative_category, i) for i in
                    os.listdir(os.path.join(pic_root_dir, negative_category))]

    pic_file_list = positive_pic + negative_pic

    for pic in pic_file_list:
        pic = os.path.join(pic_root_dir, pic)
        a_pic = cv2.imread(pic, is_color)
        a_pic = cv2.cvtColor(a_pic, cv2.COLOR_BGR2RGB)
        a_pic = cv2.resize(a_pic, pic_size, interpolation=cv2.INTER_AREA)
        img_list.append(a_pic)

    pic_vector = hog(img_list, orientations, pixels_per_cell, cells_per_block)
    label = np.array(np.repeat(1, pic_vector.shape[0]))
    label[len(positive_pic):] = 0

    x_train, x_test, y_train, y_test = train_test_split(pic_vector, label, test_size=validation_size, random_state=seed)

    # initial feature_vector.pki
    feature_vector = {
        user_id: {
            'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test
        }
    }

    with open(settings.FEATURE_VECTOR_PATH, "wb") as f:
        pickle.dump(feature_vector, f)

    with open(settings.ALGORITHM_JSON_PATH, "w") as f2:
        json.dump(algorithm_info, f2)

    return None


def execute_evaluate_algorithm(user_id, is_standard, algorithm_list):
    """
    accept the parameter, then contrast the several algorithms to svm show in boxplot by previous vector
    :param user_id:
    :param is_standard:
    :param contrast_algorithm:
    :return: None
    """

    with open(settings.FEATURE_VECTOR_PATH, "rb") as load_f:
        feature_vector = pickle.load(load_f)

    with open(settings.ALGORITHM_JSON_PATH, "r") as load_f2:
        algorithm_info = json.load(load_f2)

    algorithm_info[user_id]['data_para'].update({'is_standard': is_standard})

    with open(settings.ALGORITHM_JSON_PATH, "w") as f:
        json.dump(algorithm_info, f)

    algorithm_list.insert(0, 'SVM')
    models = {}
    if is_standard:
        algorithm_list = ['Scaler' + i for i in algorithm_list]

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
    for key in algorithm_list:
        kfold = KFold(n_splits=num_folds, random_state=seed)

        # TODO:The nums of samples should big than kfold
        cv_results = cross_val_score(models[key],
                                     feature_vector[user_id]['x_train'],
                                     feature_vector[user_id]['y_train'],
                                     cv=kfold,
                                     scoring=scoring)
        results.append(cv_results)

    # 评估算法 - 箱线图
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(algorithm_list)

    eval_pic = eval_pic_path(user_id, relative=False)
    plt.savefig(eval_pic)

    fig.clf()
    plt.close('all')

    gc.collect()
    return None


def execute_adjust_svm(user_id, c, kernel, return_dict):
    """
    get the c and kernel, return the results of svm by given parameter
    :param user_id:str
    :param c: list
    :param kernl: list
    :return: results
    """

    # TODO: optimizate the parameter
    with open(settings.FEATURE_VECTOR_PATH, "rb") as load_f:
        feature_vector = pickle.load(load_f)

    x_train, y_train = feature_vector[user_id]['x_train'], feature_vector[user_id]['y_train']
    scaler = StandardScaler().fit(x_train)
    rescaledX = scaler.transform(x_train).astype(float)
    param_grid = {
        'C': c,
        'kernel': kernel
    }
    model = SVC(gamma='scale')
    kfold = KFold(n_splits=num_folds, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X=rescaledX, y=y_train)

    cv_results = zip(grid_result.cv_results_['mean_test_score'],
                     grid_result.cv_results_['std_test_score'],
                     grid_result.cv_results_['params'])

    return_dict['best_score'] = grid_result.best_score_
    return_dict['best_params'] = grid_result.best_params_
    return_dict['cv_results'] = cv_results

    # save the best_score, best_params in algorithm_json
    with open(settings.ALGORITHM_JSON_PATH, "r") as load_f:
        algorithm_info = json.load(load_f)

    algorithm_info[user_id]['model_para'].update({
        'best_score': grid_result.best_score_,
        'best_params': grid_result.best_params_
    })

    with open(settings.ALGORITHM_JSON_PATH, "w") as f:
        json.dump(algorithm_info, f)


def execute_train_model(user_id, model_name, train_category_positive, train_category_negative, validation_size,
                        return_dict):
    model = SVMModel.objects.get(user_id=user_id, model_name=model_name)

    pic_root_dir = os.path.join(settings.MEDIA_ROOT, 'upload_images', str(user_id))
    img_list = []
    positive_pic = [os.path.join(pic_root_dir, train_category_positive, i) for i in
                    os.listdir(os.path.join(pic_root_dir, train_category_positive))]
    negative_pic = [os.path.join(pic_root_dir, train_category_negative, i) for i in
                    os.listdir(os.path.join(pic_root_dir, train_category_negative))]

    pic_file_list = positive_pic + negative_pic

    for pic in pic_file_list:
        pic = os.path.join(pic_root_dir, pic)
        a_pic = cv2.imread(pic, model.is_color)
        a_pic = cv2.resize(a_pic, eval(model.pic_size), interpolation=cv2.INTER_AREA)
        img_list.append(a_pic)

    pic_vector = hog(img_list, model.orientations, eval(model.pixels_per_cell), eval(model.cells_per_block))
    label = np.array(np.repeat(1, pic_vector.shape[0]))
    label[len(positive_pic):] = 0

    x_train, x_test, y_train, y_test = train_test_split(pic_vector, label, test_size=validation_size, random_state=seed)

    the_path = path.join(settings.MEDIA_ROOT, 'upload_models', str(user_id), model_name + '.model')
    with open(the_path, 'rb') as model_f:
        svm_model = joblib.load(model_f)

    if model.is_standard:
        scaler = StandardScaler().fit(x_train)

        rescaledX = scaler.transform(x_train)
        svm_model.fit(X=rescaledX, y=y_train)

        rescaled_validationX = scaler.transform(x_test)
        predictions = svm_model.predict(rescaled_validationX)

        svm_model.fit(rescaled_validationX, y_test)

    else:
        svm_model.fit(X=x_train, y=y_train)
        predictions = model.predict(x_test)
        svm_model.fit(x_test, y_test)

    return_dict['accuracy_score'] = accuracy_score(y_test, predictions)
    return_dict['confusion_matrix'] = confusion_matrix(y_test, predictions)
    return_dict['classification_report'] = classification_report(y_test, predictions)

    model.train_num += 1
    model.recently_accuracy_score = accuracy_score(y_test, predictions)
    model.save()

    with open(the_path, 'wb') as f:
        joblib.dump(svm_model, f)


def execute_adjust_ensemble(user_id, C, kernel, ensemble_learning, n_estimators, return_dict):
    with open(settings.FEATURE_VECTOR_PATH, "rb") as load_f:
        feature_vector = pickle.load(load_f)

    x_train, y_train = feature_vector[str(user_id)]['x_train'], feature_vector[str(user_id)]['y_train']

    scaler = StandardScaler().fit(x_train)
    rescaledX = scaler.transform(x_train)
    param_grid = {'n_estimators': n_estimators}

    cart = SVC(gamma='scale', C=C, kernel=kernel, probability=True)

    if ensemble_learning == 'BaggingClassifier':
        model = BaggingClassifier(base_estimator=cart, random_state=seed)
    elif ensemble_learning == 'AdaBoostClassifier':
        model = AdaBoostClassifier(base_estimator=cart, random_state=seed)
    else:
        return False

    kfold = KFold(n_splits=num_folds, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X=rescaledX, y=y_train)

    cv_results = zip(grid_result.cv_results_['mean_test_score'],
                     grid_result.cv_results_['std_test_score'],
                     grid_result.cv_results_['params'])
    return_dict['best_score'] = grid_result.best_score_
    return_dict['best_params'] = grid_result.best_params_
    return_dict['cv_results'] = cv_results

    # save the best_score, best_params in algorithm_json
    with open(settings.ALGORITHM_JSON_PATH, "r") as load_f:
        algorithm_info = json.load(load_f)

    algorithm_info[str(user_id)]['ensemble_para'].update({
        'ensemble_learning': ensemble_learning,
        'n_estimators': grid_result.best_params_['n_estimators'],
        'best_score': grid_result.best_score_,
    })

    with open(settings.ALGORITHM_JSON_PATH, "w") as f:
        json.dump(algorithm_info, f)


def execute_cat_identification(user_id, model_name, show_probility, return_dict):
    model_db = SVMModel.objects.get(user_id=user_id, model_name=model_name)

    the_path = path.join(settings.MEDIA_ROOT, 'upload_models', str(user_id), model_name + '.model')
    with open(the_path, 'rb') as model_f:
        model = joblib.load(model_f)

    pic_root_dir = path.join(settings.MEDIA_ROOT, 'predict_images',
                             str(user_id))
    pic_list = os.listdir(os.path.join(pic_root_dir))

    img_list = []
    for pic in pic_list:
        pic = os.path.join(pic_root_dir, pic)
        a_pic = cv2.imread(pic, model_db.is_color)
        a_pic = cv2.resize(a_pic, eval(model_db.pic_size), interpolation=cv2.INTER_AREA)
        img_list.append(a_pic)

    pic_vector = hog(img_list, model_db.orientations, eval(model_db.pixels_per_cell), eval(model_db.cells_per_block))

    if model_db.is_standard:
        scaler = StandardScaler().fit(pic_vector)
        pic_vector = scaler.transform(pic_vector)

    if show_probility:
        predictions = (model.predict_proba(pic_vector))[:, 1]
    else:
        predictions = model.predict(pic_vector)

    return_dict['predictions'] = predictions
