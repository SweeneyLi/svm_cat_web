from django.shortcuts import render, reverse, redirect
from django.urls import reverse_lazy
from django.views.generic import CreateView, ListView, DetailView, FormView
from django.utils.safestring import mark_safe

from .forms import *
from .functions import *
from .algorithm_conf import *
from .models import SVMModel, ModelTrainLog

from sklearn.externals import joblib
import multiprocessing as mp
from os import path, mkdir
import json
import shutil


class template(FormView):
    form_class = PrepareDataForm

    def get_form(self, form_class=None):
        pass

    def get(self, request, *args, **kwargs):
        pass

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form, **kwargs)
        else:
            return self.form_invalid(form, **kwargs)

    def form_invalid(self, form, **kwargs):
        return render(self.request, 'algorithm/choose_pic_category.html',
                      {'form': form, 'error_message': form.errors
                       })

    def form_valid(self, form, **kwargs):
        pass


class PrepareDataView(FormView):
    form_class = PrepareDataForm

    def get_form(self, form_class=None):
        form_class = self.get_form_class()
        return form_class(self.request.user.id, **self.get_form_kwargs())

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(request, 'algorithm/choose_pic_category.html',
                      {'form': form})

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form, **kwargs)
        else:
            return self.form_invalid(form, **kwargs)

    def form_invalid(self, form, **kwargs):
        return render(self.request, 'algorithm/choose_pic_category.html',
                      {'form': form,
                       'error_message': form.errors
                       })

    def form_valid(self, form, **kwargs):

        user_id = str(self.request.user.id)
        test_pic = form.files['test_pic']
        validation_size = float(form.data['validation_size'])
        test_category_positive = eval(form.data['test_category_positive'])
        test_category_negative = eval(form.data['test_category_negative'])

        #  save the test_pic
        pic_name = test_pic.name
        saved_pic = saved_pic_path(user_id)

        with open(saved_pic, 'wb+') as destination:
            for chunk in test_pic.chunks():
                destination.write(chunk)

        # save the test_pic_name in json
        with open(settings.ALGORITHM_JSON_PATH, "r") as load_f:
            algorithm_info = json.load(load_f)

        # algorithm_info_json initial
        if user_id not in algorithm_info.keys():
            algorithm_info[user_id] = {}

        algorithm_info_keys = algorithm_info[user_id].keys()
        for key in ['pic_para', 'data_para', 'model_para', 'ensemble_para']:
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

        with open(settings.ALGORITHM_JSON_PATH, 'w') as f:
            json.dump(algorithm_info, f)

        return redirect(reverse('alogrithm:hog_pic'))


class HOGPicView(FormView):
    form_class = HOGPicForm

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(request, 'algorithm/hog_pic.html', {'form': form})

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form, **kwargs)
        else:
            return self.form_invalid(form, **kwargs)

    def form_invalid(self, form, **kwargs):
        return render(self.request, 'algorithm/hog_pic.html',
                      {'form': form, 'error_message': form.errors})

    def form_valid(self, form, **kwargs):
        pic_size = eval(form.data['pic_size'])
        orientations = int(form.data['orientations'])
        pixels_per_cell = eval(form.data['pixels_per_cell'])
        cells_per_block = eval(form.data['cells_per_block'])
        is_color = True if 'is_color' in form.data else False
        user_id = str(self.request.user.id)

        proc = mp.Process(target=execute_hog_pic,
                          args=(pic_size, orientations, pixels_per_cell, cells_per_block, is_color, user_id))
        proc.daemon = True
        proc.start()
        proc.join()

        # get the saved png to show in page
        hog_pic = hog_pic_path(user_id)

        return render(self.request, 'algorithm/hog_pic.html',
                      {'form': form,
                       'hog_pic': hog_pic,
                       })


class EvaluateAlgorithmView(FormView):
    form_class = EvaluateAlgoritmForm

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        # hog the picture to feature vector and sava as feature_vector.pkl
        proc = mp.Process(target=get_pic_vector, args=(str(request.user.id)))
        proc.daemon = True
        proc.start()
        proc.join()

        return render(request, 'algorithm/contrast_algorithm.html',
                      {'form': form})

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        user_id = str(self.request.user.id)

        algorithm_list = self.request.POST.getlist('algorithm_list')
        is_standard = True if self.request.POST.get('is_standard') else False
        proc = mp.Process(target=execute_evaluate_algorithm, args=(user_id, is_standard, algorithm_list))
        proc.daemon = True
        proc.start()
        proc.join()

        eval_pic = eval_pic_path(user_id)

        return render(self.request, 'algorithm/contrast_algorithm.html',
                      {'form': form,
                       'eval_pic': eval_pic,
                       })


class AdjustSVMView(FormView):
    form_class = SVMParameterForm

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(request, 'algorithm/adjust_svm.html', {'form': form})

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form, **kwargs)
        else:
            return self.form_invalid(form, **kwargs)

    def form_invalid(self, form, **kwargs):
        return render(self.request, 'algorithm/adjust_svm.html',
                      {'form': form, 'error_message': form.errors
                       })

    def form_valid(self, form, **kwargs):
        form = self.get_form()
        user_id = str(self.request.user.id)
        C = form.data['C'].split(',')
        C = list(map(lambda a: float(a), C))
        kernel = self.request.POST.getlist('kernel')

        manager = mp.Manager()
        return_dict = manager.dict()
        proc = mp.Process(target=execute_adjust_svm, args=(user_id, C, kernel, return_dict))
        proc.daemon = True
        proc.start()
        proc.join()

        return render(self.request, 'algorithm/adjust_svm.html',
                      {'form': form,
                       'results': return_dict})


def adjust_svm(request):
    if request.method == 'POST':
        user_id = str(request.user.id)
        C = request.POST.get('C').split(',')
        C = list(map(lambda a: float(a), C))
        kernel = request.POST.getlist('kernel')

        manager = mp.Manager()
        return_dict = manager.dict()
        proc = mp.Process(target=execute_adjust_svm, args=(user_id, C, kernel, return_dict))
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


def adjust_ensemble_learning(request):
    if request.method == 'POST':
        user_id = request.user.id
        C = float(request.POST.get('C'))
        kernel = request.POST.get('kernel')
        n_estimators = eval(request.POST.get('n_estimators'))
        ensemble_learning = request.POST.get('ensemble_learning')

        manager = mp.Manager()
        return_dict = manager.dict()
        proc = mp.Process(target=execute_adjust_ensemble,
                          args=(user_id, C, kernel, ensemble_learning, n_estimators, return_dict))
        proc.daemon = True
        proc.start()
        proc.join()

        ensemble_params_form = EnsembleParamsForm(user_id, request.POST)
        return render(request, 'algorithm/adjust_ensemble_learning.html',
                      {'ensemble_params_form': ensemble_params_form,
                       'results': return_dict})
    else:
        ensemble_params_form = EnsembleParamsForm(request.user.id)
        return render(request, 'algorithm/adjust_ensemble_learning.html',
                      {'ensemble_params_form': ensemble_params_form})


class ModelCreateView(CreateView):
    template_name = 'algorithm/create_svm_model.html'
    model = SVMModel
    fields = ['model_name', 'pic_size', 'orientations',
              'pixels_per_cell', 'cells_per_block', 'is_color',
              'is_standard', 'C', 'kernel', 'ensemble_learning',
              'n_estimators']

    def get_success_url(self):
        return reverse_lazy('alogrithm:train_svm_model')

    def get_form_kwargs(self):
        kwargs = super(ModelCreateView, self).get_form_kwargs()

        with open(settings.ALGORITHM_JSON_PATH, "r") as load_f:
            algorithm_info = json.load(load_f)
        user_id = str(self.request.user.id)

        kwargs['initial']['model_name'] = algorithm_info[user_id]['data_para'].get('category_positive',
                                                                                   "svm") + '_model'
        kwargs['initial']['pic_size'] = algorithm_info[user_id]['pic_para'].get('pic_size', "(194, 259)")
        kwargs['initial']['orientations'] = algorithm_info[user_id]['pic_para'].get('orientations', 9)
        kwargs['initial']['pixels_per_cell'] = algorithm_info[user_id]['pic_para'].get('pixels_per_cell', "(8, 8)")
        kwargs['initial']['cells_per_block'] = algorithm_info[user_id]['pic_para'].get('cells_per_block', "(3, 3)")
        kwargs['initial']['is_color'] = algorithm_info[user_id]['pic_para'].get('is_color', True)
        kwargs['initial']['is_standard'] = algorithm_info[user_id]['data_para'].get('is_standard', True)

        model_best_params = algorithm_info[user_id]['model_para'].get('best_params', {'C': 2.0, 'kernel': 'sigmoid'})
        kwargs['initial']['C'] = model_best_params['C']
        kwargs['initial']['kernel'] = model_best_params['kernel']

        kwargs['initial']['ensemble_learning'] = algorithm_info[user_id]['ensemble_para'].get('ensemble_learning',
                                                                                              'None')
        kwargs['initial']['n_estimators'] = algorithm_info[user_id]['ensemble_para'].get('n_estimators', 0)

        return kwargs

    def form_valid(self, form):
        user_id = self.request.user.id

        # judge the same model_name
        if SVMModel.objects.filter(user_id=user_id, model_name=form.data['model_name']).exists():
            form.add_error(None, "The model_name is created!")
            return super().form_invalid(form)

        form.instance.user_id = user_id

        # save the model in local
        svm_model = SVC(C=float(form.data['C']), kernel=form.data['kernel'], probability=True)

        if form.data['ensemble_learning'] == 'BaggingClassifier':
            svm_model = BaggingClassifier(base_estimator=svm_model, n_estimators=form.data['n_estimators'])
        elif form.data['AdaBoostClassifier'] == 'AdaBoostClassifier':
            svm_model = AdaBoostClassifier(base_estimator=svm_model, n_estimators=form.data['n_estimators'])

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

        train_log_form = TrainLogForm(request.user.id)

        # TODO：format the result in page
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
            filter(user_id=self.request.user.id).order_by('recently_accuracy_score', '-update_time')


class ModelDetail(DetailView):
    model = SVMModel
    Context_object_name = 'model_detail'

    template_name = 'algorithm/model_detail.html'


def cat_identification(request):
    user_id = request.user.id
    if request.method == 'POST':
        model_name = eval(request.POST.get('model_name'))['model_name']

        model_db = SVMModel.objects.get(user_id=user_id, model_name=model_name)
        if model_db.train_num == 0:
            cat_identification_form = CatIdentificationForm(request.user.id)
            return render(request, 'algorithm/cat_identification.html',
                          {'cat_identification_form': cat_identification_form,
                           'error_message': 'The trained model could predict, please train it'})

        files = request.FILES.getlist('file')
        show_probility = request.POST.get('show_probility')

        # save the files
        saved_pic_root = path.join(settings.MEDIA_ROOT, 'predict_images',
                                   str(user_id))
        if path.exists(saved_pic_root):
            shutil.rmtree(saved_pic_root)
        mkdir(saved_pic_root)

        for f in files:
            pic_name = f.name
            with open(os.path.join(saved_pic_root, pic_name), 'wb+') as destination:
                for chunk in f.chunks():
                    destination.write(chunk)

        manager = mp.Manager()
        return_dict = manager.dict()
        proc = mp.Process(target=execute_cat_identification, args=(
            user_id, model_name, show_probility, return_dict
        ))
        proc.daemon = True
        proc.start()
        proc.join()

        with open(settings.ALGORITHM_JSON_PATH, "r") as load_f:
            algorithm_info = json.load(load_f)

        cat_identification_form = CatIdentificationForm(request.user.id)
        return render(request, 'algorithm/cat_identification.html',
                      {'cat_identification_form': cat_identification_form,
                       'result': return_dict
                       })
    else:
        cat_identification_form = CatIdentificationForm(request.user.id)
        return render(request, 'algorithm/cat_identification.html',
                      {'cat_identification_form': cat_identification_form})
