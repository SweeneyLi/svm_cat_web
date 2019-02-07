from django.shortcuts import render, reverse, redirect
from django.urls import reverse_lazy
from django.views.generic import CreateView, ListView, DetailView

from .forms import *
from .functions import *
from .models import SVMModel, ModelTrainLog
from sklearn.externals import joblib
import multiprocessing as mp
from os import path, mkdir
import json
import pickle


def choose_pic_category(request):
    # TODO: add the loading html
    user_id = str(request.user.id)

    if request.method == 'POST':

        # TODO: judege choose_pic_category_form.is_valid()

        # get the parameter
        test_pic = request.FILES.get('test_pic')
        validation_size = float(request.POST.get('validation_size'))
        test_category_positive = eval(request.POST.get('test_category_positive'))
        test_category_negative = eval(request.POST.get('test_category_negative'))

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
        if user_id not in algorithm_info.keys():
            algorithm_info[user_id] = {}

        algorithm_info_keys = algorithm_info[user_id].keys()
        for key in ['pic_para', 'data_para', 'model_para']:
            if key not in algorithm_info_keys:
                algorithm_info[user_id][key] = {}

        algorithm_info[user_id]['pic_para'].update({'test_pic': pic_name})
        algorithm_info[user_id]['data_para'].update({
            'category_positive': test_category_positive['category'],
            'category_negative': test_category_negative['category'],
            'num_category_positive': test_category_positive['num_category'],
            'num_category_negative': test_category_negative['num_category'],
            'validation_size': validation_size,
        })

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
        pic_size = eval(request.POST.get('pic_size'))
        orientations = int(request.POST.get('orientations'))
        pixels_per_cell = eval(request.POST.get('pixels_per_cell'))
        cells_per_block = eval(request.POST.get('cells_per_block'))
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

        return render(request, 'algorithm/hog_pic.html',
                      {'hog_pic_form': hog_pic_form,
                       'hog_picture': relative_pic_path,
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


class ModelCreateView(CreateView):
    template_name = 'algorithm/create_svm_model.html'
    model = SVMModel
    fields = ['model_name', 'pic_size', 'orientations',
              'pixels_per_cell', 'cells_per_block', 'is_color',
              'is_standard', 'C', 'kernel']

    def get_success_url(self):
        return reverse_lazy('alogrithm:train_svm_model')

    def get_form_kwargs(self):
        kwargs = super(ModelCreateView, self).get_form_kwargs()

        with open(settings.ALGORITHM_JSON_PATH, "r") as load_f:
            algorithm_info = json.load(load_f)
        user_id = str(self.request.user.id)

        kwargs['initial']['model_name'] = algorithm_info[user_id]['data_para']. \
                                              get('category_positive', "svm") + '_model'

        kwargs['initial']['pic_size'] = algorithm_info[user_id]['pic_para'].get('pic_size', "(194, 259)")
        kwargs['initial']['orientations'] = algorithm_info[user_id]['pic_para'].get('orientations', 9)
        kwargs['initial']['pixels_per_cell'] = algorithm_info[user_id]['pic_para'].get('pixels_per_cell', "(8, 8)")
        kwargs['initial']['cells_per_block'] = algorithm_info[user_id]['pic_para'].get('cells_per_block', "(3, 3)")
        kwargs['initial']['is_color'] = algorithm_info[user_id]['pic_para'].get('is_color', True)
        kwargs['initial']['is_standard'] = algorithm_info[user_id]['data_para'].get('is_standard', True)

        best_params = algorithm_info[user_id]['model_para'].get('best_params', {'C': 2.0, 'kernel': 'sigmoid'})
        kwargs['initial']['C'] = best_params['C']
        kwargs['initial']['kernel'] = best_params['kernel']

        return kwargs

    def form_valid(self, form):
        user_id = self.request.user.id

        # judge the same model_name
        if SVMModel.objects.filter(user_id=user_id, model_name=form.data['model_name']).exists():
            form.add_error(None, "The model_name is created!")
            return super().form_invalid(form)

        form.instance.user_id = user_id

        # save the model in local
        svm_model = SVC(C=float(form.data['C']), kernel=form.data['kernel'])

        the_dir = path.join(settings.MEDIA_ROOT, 'upload_models', str(user_id))
        if not path.exists(the_dir):
            mkdir(the_dir)
        filename = form.data['model_name'] + '.model'
        the_path = path.join(the_dir, filename)
        with open(the_path, 'wb') as model_f:
            joblib.dump(svm_model, model_f)

        return super().form_valid(form)


def train_svm_model(request):
    if request.method == 'POST':
        user_id = request.user.id
        model_name = eval(request.POST.get('model_name'))['model_name']
        train_category_positive_dict = eval(request.POST.get('train_category_positive'))
        train_category_negative_dict = eval(request.POST.get('train_category_negative'))

        train_category_positive = train_category_positive_dict['category']
        positive_num = train_category_positive_dict['num_category']
        train_category_negative = train_category_negative_dict['category']
        negative_num = train_category_negative_dict['num_category']
        validation_size = float(request.POST.get('validation_size'))

        # train the model
        manager = mp.Manager()
        return_dict = manager.dict()
        proc = mp.Process(target=execute_train_model, args=(
            user_id, model_name, train_category_positive, train_category_negative, validation_size, return_dict
        ))
        proc.daemon = True
        proc.start()
        proc.join()

        train_log = ModelTrainLog(user_id=user_id, model_name=model_name,
                                  train_category_positive=train_category_positive,
                                  positive_num=positive_num,
                                  train_category_negative=train_category_negative,
                                  negative_num=negative_num,
                                  validation_size=validation_size,
                                  accuracy_score=return_dict['accuracy_score'])

        train_log.save()
        svm_model = SVMModel.objects.get(user_id=user_id, model_name=model_name)
        svm_model.train_num += 1
        svm_model.save()

        train_log_form = TrainLogForm(request.user.id)
        return render(request, 'algorithm/train_svm_model.html',
                      {'train_log_form': train_log_form,
                       'result': return_dict})
    else:
        train_log_form = TrainLogForm(request.user.id)
        return render(request, 'algorithm/train_svm_model.html',
                      {'train_log_form': train_log_form})


class ModelListView(ListView):
    context_object_name = 'model_list'

    template_name = 'algorithm/model_list.html'

    def get_queryset(self):
        return SVMModel.objects.all(). \
            filter(user_id=self.request.user.id).order_by('accuracy_score', '-update_time')


class ModelDetail(DetailView):
    model = SVMModel
    Context_object_name = 'model_detail'

    template_name = 'algorithm/model_detail.html'

    # def get_queryset(self):
    #     return SVMModel.objects.get(id=int(self.kwargs['pk']))
