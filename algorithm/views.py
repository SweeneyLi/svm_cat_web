from django.shortcuts import render, reverse, redirect
from django.conf import settings

from .forms import HOGPicForm, ChoosePicCategoryForm
from .tools import cal_hog_time, str_to_tuple, get_pic_vector
from .algorithm_conf import *

from os import path
from skimage import feature, exposure
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split


def choose_pic_category(request):
    user_id = str(request.user.id)

    if request.method == 'POST':

        # TODO: judege choose_pic_category_form.is_valid()

        # get the parameter
        test_pic = request.FILES.get('test_pic')
        test_category_positive = request.POST.get('test_category_positive')
        test_category_negative = request.POST.get('test_category_negative')

        test_category_positive = test_category_positive.split('\'')[-2]
        test_category_negative = test_category_negative.split('\'')[-2]

        #  save the test_pic
        pic_name = request.FILES.get('test_pic').name
        saved_pic_path = path.join(settings.MEDIA_ROOT, 'algorithm', 'hog_picture',
                                   user_id + '_' + 'hog_test_pic.jpg')
        with open(saved_pic_path, 'wb+') as destination:
            for chunk in test_pic.chunks():
                destination.write(chunk)

        # save the test_pic_name in json
        with open(settings.ALGORITHM_JSON_PATH, "r") as load_f:
            algorithm_info = json.load(load_f)

        now = (datetime.now() + timedelta(hours=8)).strftime("%Y-%m-%d_%H:%M:%S.")
        algorithm_info[user_id] = {'test_pic': pic_name,
                                   'pic_para': {},
                                   'test_category': {
                                       'positive': test_category_positive,
                                       'negative': test_category_negative
                                   },
                                   'update_time': now
                                   }

        with open(settings.ALGORITHM_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(algorithm_info, f)

        return redirect(reverse('alogrithm:hog_pic'))
    else:

        choose_pic_category_form = ChoosePicCategoryForm(user_id=user_id)
        return render(request, 'algorithm/choose_pic_category.html',
                      {'choose_pic_category_form': choose_pic_category_form})


def hog_pic(request):
    if request.method == 'POST':

        # get the parameter
        pic_size = request.POST.get('pic_size')
        orientations = int(request.POST.get('orientations'))
        pixels_per_cell = request.POST.get('pixels_per_cell')
        cells_per_block = request.POST.get('cells_per_block')
        is_color = request.POST.get('is_color')
        user_id = str(request.user.id)

        # TODO:form validation
        # # 判断参数，是否窗口尺寸大于图片尺寸
        # if pixels_per_cell[0] * cells_per_block[0] >= pic_size[0] or pixels_per_cell[1] * cells_per_block[1] >= \
        #         pic_size[1]:
        #     return False

        # 取出json中的test_pic, 放入图片参数
        with open(settings.ALGORITHM_JSON_PATH, "r") as load_f:
            algorithm_info = json.load(load_f)

        test_pic = algorithm_info[user_id]['test_pic']
        algorithm_info[user_id]['pic_para'] = {
            'pic_size': pic_size,
            'orientations': orientations,
            'pixels_per_cell': pixels_per_cell,
            'cells_per_block': cells_per_block,
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
        if is_color:
            img = cv2.imread(saved_pic_path, 1)
        else:
            img = cv2.imread(saved_pic_path, 0)
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
                                     str(request.user.id) + '_hog_picture.png')
        plt.savefig(hog_picture_path)

        # get the saved png to show in page
        relative_pic_path = hog_picture_path.replace(settings.BASE_DIR, '')
        hog_pic_form = HOGPicForm(request.POST)

        # TODO: meet the memory problem when calcuate the hog time
        hog_time = cal_hog_time(img, orientations, pixels_per_cell, cells_per_block)
        return render(request, 'algorithm/hog_pic.html',
                      {'hog_pic_form': hog_pic_form, 'hog_picture': relative_pic_path, 'hog_time': hog_time})

    else:
        hog_pic_form = HOGPicForm()
        return render(request, 'algorithm/hog_pic.html', {'hog_pic_form': hog_pic_form})


def create_model(request):
    pic_vector, label = get_pic_vector(str(request.user.id))
    X_train, X_test, y_train, y_test = train_test_split(pic_vector, label, test_size=validation_size, random_state=seed)
    return render(request, 'algorithm/create_model.html', {'a': y_train, 'b': y_test})
