from django.shortcuts import render, reverse, redirect

from .forms import *
from .functions import *

from os import path
import json
from datetime import datetime, timedelta
import multiprocessing as mp


def choose_pic_category(request):
    # TODO: add the loading html
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




        # algorithm_info_json initial
        algorithm_info[user_id] = {'pic_para': {
            'test_pic': pic_name
        }, 'data_para': {
            'category_positive': test_category_positive,
            'category_negative': test_category_negative,
            'validation_size': validation_size,
        },
            # 'update_time': now
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
        is_color = True if request.POST.get('is_color') else False
        user_id = str(request.user.id)

        # execute_hog_pic(pic_size, orientations, pixels_per_cell, cells_per_block, is_color, user_id)

        proc = mp.Process(target=execute_hog_pic,
                          args=(pic_size, orientations, pixels_per_cell, cells_per_block, is_color, user_id))
        proc.daemon = True
        proc.start()
        proc.join()

        # get the saved png to show in page
        relative_pic_path = path.join('/media', 'algorithm', 'hog_picture',
                                      user_id + '_hog_picture.png')
        hog_pic_form = HOGPicForm(request.POST)

        with open(settings.ALGORITHM_JSON_PATH, "r") as load_f:
            algorithm_info = json.load(load_f)

            # hog_time = algorithm_info[user_id]['pic_para']['hog_time']

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
        user_id = str(request.user.id)

        # TODO: complete the choices of contrast_alorgitm
        # contrast_algorithm = request.POST.getlist('contrast_algorithm')
        contrast_algorithm = False

        is_standard = True if request.POST.get('is_standard') else False

        proc = mp.Process(target=execute_contrast_algorithm, args=(user_id, is_standard, contrast_algorithm))
        proc.daemon = True
        proc.start()
        proc.join()

        relative_pic_path = path.join('/media', 'algorithm', 'hog_picture',
                                      user_id + '_contrast_algorithm.png')
        contrast_algorithm_form = ContrastAlgorithmForm(request.POST)

        return render(request, 'algorithm/contrast_algorithm.html',
                      {'contrast_algorithm_form': contrast_algorithm_form,
                       'algorithm_picture': relative_pic_path,
                       })
    else:
        get_pic_vector(str(request.user.id))
        contrast_algorithm_form = ContrastAlgorithmForm()
        return render(request, 'algorithm/contrast_algorithm.html',
                      {'contrast_algorithm_form': contrast_algorithm_form})


def adjust_svm(request):
    if request.method == 'POST':
        user_id = str(request.user.id)
        c = request.POST.get('c').split(',')
        c = list(map(lambda a: float(a), c))
        kernel = request.POST.getlist('kernel')

        manager = mp.Manager()
        return_dict = manager.dict()
        proc = mp.Process(target=execute_adjust_svm, args=(user_id, c, kernel, return_dict))
        proc.daemon = True
        proc.start()
        proc.join()

        svm_parameter_form = SVMParameterForm(request.POST)
        return render(request, 'algorithm/adjust_svm.html',
                      {'svm_parameter_form': svm_parameter_form,
                       'results': return_dict})
    else:
        svm_parameter_form = SVMParameterForm()
        return render(request, 'algorithm/adjust_svm.html', {'svm_parameter_form': svm_parameter_form})
