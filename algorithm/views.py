from django.shortcuts import render, reverse, redirect

from .forms import HOGPicForm, ChoosePicCategoryForm, ContrastAlgorithmForm
from .tools import *

from os import path
import json
from datetime import datetime, timedelta
import multiprocessing as mp

def choose_pic_category(request):
    user_id = str(request.user.id)

    if request.method == 'POST':

        # TODO: judege choose_pic_category_form.is_valid()

        # get the parameter
        test_pic = request.FILES.get('test_pic')
        validation_size = float(request.POST.get('validation_size'))
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

        # json initial
        algorithm_info[user_id] = {'pic_para': {
            'test_pic': pic_name
        }, 'model_para': {
            'category_positive': test_category_positive,
            'category_negative': test_category_negative,
            'validation_size': validation_size,
        }, 'update_time': now
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

        # TODO:form validation
        # # 判断参数，是否窗口尺寸大于图片尺寸
        # if pixels_per_cell[0] * cells_per_block[0] >= pic_size[0] or pixels_per_cell[1] * cells_per_block[1] >= \
        #         pic_size[1]:
        #     return False

        # get the parameter
        pic_size = request.POST.get('pic_size')
        orientations = int(request.POST.get('orientations'))
        pixels_per_cell = request.POST.get('pixels_per_cell')
        cells_per_block = request.POST.get('cells_per_block')
        is_color = request.POST.get('is_color')
        user_id = str(request.user.id)

        execute_hog_pic(pic_size, orientations, pixels_per_cell,cells_per_block, is_color, user_id)

        proc = mp.Process(target=execute_hog_pic, args=(pic_size, orientations, pixels_per_cell,cells_per_block, is_color, user_id))
        proc.daemon = True
        proc.start()
        proc.join()


        # get the saved png to show in page
        relative_pic_path = path.join('/media', 'algorithm', 'hog_picture',
                                      user_id + '_hog_picture.png')
        hog_pic_form = HOGPicForm(request.POST)

        # TODO: meet the memory problem when calcuate the hog time
        # hog_time = cal_hog_time(img, orientations, pixels_per_cell, cells_per_block)

        return render(request, 'algorithm/hog_pic.html',
                      {'hog_pic_form': hog_pic_form,
                       'hog_picture': relative_pic_path,
                       # 'hog_time': hog_time
                       })

    else:
        hog_pic_form = HOGPicForm()
        return render(request, 'algorithm/hog_pic.html', {'hog_pic_form': hog_pic_form})


def contrast_algorithm(request):
    if request.method == 'POST':
        X_train, X_test, y_train, y_test = get_pic_vector(str(request.user.id))


    else:
        contrast_algorithm_form = ContrastAlgorithmForm()
        return render(request, 'algorithm/', {'contrast_algorithm_form': contrast_algorithm_form})


def create_model(request):
    X_train, X_test, y_train, y_test = get_pic_vector(str(request.user.id))

    return render(request, 'algorithm/create_model.html', {'a': y_train, 'b': y_test})



