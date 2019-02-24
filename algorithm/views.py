from django.shortcuts import render, reverse, redirect
from django.views.generic import CreateView, ListView, DetailView, FormView, DeleteView, View
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


# turn step to specific url
def Step(request, pk):
    pk = int(pk)
    if 0 < pk < 7:
        return redirect(step_dict[pk]['url'])
    elif pk <= 0:
        return redirect(step_dict[1]['url'])
    else:
        return redirect(step_dict[len(step_dict)]['url'])


# template
class template(FormView):
    form_class = PrepareDataForm

    def get_form(self, form_class=None):
        pass

    def get(self, request, *args, **kwargs):
        form = None

        return render(request, 'algorithm/model_form.html',
                      {'form': form,
                       'step': 2,
                       'title': step_dict[2]['name'],
                       'remark': mark_safe('<button>1233</button>'),
                       'picture': None
                       })

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form, **kwargs)
        else:
            return self.form_invalid(form, **kwargs)

    def form_invalid(self, form, **kwargs):
        return render(self.request, 'algorithm/choose_pic_category.html',
                      {'form': form, 'message': form.errors
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
        return render(request, 'algorithm/model_form.html',
                      {'form': form,
                       'step': 1,
                       'title': step_dict[1]['name'],
                       'remark': mark_safe('<button>1233</button>'),
                       'picture': None
                       })

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form, **kwargs)
        else:
            return self.form_invalid(form, **kwargs)

    def form_invalid(self, form, **kwargs):
        return render(self.request, 'algorithm/model_form.html',
                      {'form': form,
                       'step': 1,
                       'title': step_dict[1]['name'],
                       'remark': mark_safe('<button>1233</button>'),
                       'picture': None,
                       'message': form.errors
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

        return redirect('alogrithm:hog_pic')


class HOGPicView(FormView):
    form_class = HOGPicForm

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(request, 'algorithm/model_form.html',
                      {'form': form,
                       'step': 2,
                       'title': step_dict[2]['name'],
                       'remark': mark_safe('<button>1233</button>'),
                       'picture': None
                       })

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form, **kwargs)
        else:
            return self.form_invalid(form, **kwargs)

    def form_invalid(self, form, **kwargs):

        return render(self.request, 'algorithm/model_form.html',
                      {'form': form,
                       'step': 2,
                       'title': step_dict[2]['name'],
                       'remark': mark_safe('<button>1233</button>'),
                       'picture': None,
                       'message': form.errors
                       })

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

        # return render(self.request, 'algorithm/hog_pic.html',
        #               {'form': form,
        #                'hog_pic': hog_pic,
        #                })
        return render(self.request, 'algorithm/model_form.html',
                      {'form': form,
                       'step': 2,
                       'title': step_dict[2]['name'],
                       'remark': mark_safe('<button>1233</button>'),
                       'picture': hog_pic
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

        return render(request, 'algorithm/model_form.html',
                      {'form': form,
                       'step': 3,
                       'title': step_dict[3]['name'],
                       'remark': mark_safe('<button>1233</button>'),
                       'picture': None
                       })

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

        return render(request, 'algorithm/model_form.html',
                      {'form': form,
                       'step': 3,
                       'title': step_dict[3]['name'],
                       'remark': mark_safe('<button>1233</button>'),
                       'picture': eval_pic
                       })


class AdjustSVMView(FormView):
    form_class = SVMParameterForm

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(request, 'algorithm/model_form.html',
                      {'form': form,
                       'step': 4,
                       'title': step_dict[4]['name'],
                       'remark': mark_safe('<button>1233</button>'),
                       })

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form, **kwargs)
        else:
            return self.form_invalid(form, **kwargs)

    def form_invalid(self, form, **kwargs):
        return render(self.request, 'algorithm/model_form.html',
                      {'form': form,
                       'step': 4,
                       'title': step_dict[4]['name'],
                       'remark': mark_safe('<button>1233</button>'),
                       'message': form.errors
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

        return render(self.request, 'algorithm/model_form.html',
                      {'form': form,
                       'step': 4,
                       'title': step_dict[4]['name'],
                       'remark': mark_safe('<button>1233</button>'),
                       'results': return_dict
                       })


class AdjustEnsembleLearningView(FormView):
    form_class = EnsembleParamsForm

    def get_form(self, form_class=None):
        form_class = self.get_form_class()
        return form_class(self.request.user.id, **self.get_form_kwargs())

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(self.request, 'algorithm/model_form.html',
                      {'form': form,
                       'step': 5,
                       'title': step_dict[5]['name'],
                       'remark': mark_safe('<button>1233</button>'),
                       })

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form, **kwargs)
        else:
            return self.form_invalid(form, **kwargs)

    def form_invalid(self, form, **kwargs):
        return render(self.request, 'algorithm/model_form.html',
                      {'form': form,
                       'step': 5,
                       'title': step_dict[5]['name'],
                       'remark': mark_safe('<button>1233</button>'),
                       'message': form.errors
                       })

    def form_valid(self, form, **kwargs):
        user_id = self.request.user.id
        C = float(form.data['C'])
        kernel = self.request.POST.get('kernel')
        n_estimators = eval(form.data['n_estimators'])
        ensemble_learning = form.data['ensemble_learning']

        manager = mp.Manager()
        return_dict = manager.dict()
        proc = mp.Process(target=execute_adjust_ensemble,
                          args=(user_id, C, kernel, ensemble_learning, n_estimators, return_dict))
        proc.daemon = True
        proc.start()
        proc.join()

        return render(self.request, 'algorithm/model_form.html',
                      {'form': form,
                       'step': 5,
                       'title': step_dict[5]['name'],
                       'remark': mark_safe('<button>1233</button>'),
                       'results': return_dict
                       })


class ModelCreateView(CreateView):
    template_name = 'algorithm/model_form.html'
    model = SVMModel
    fields = ['model_name', 'comment', 'pic_size', 'orientations',
              'pixels_per_cell', 'cells_per_block', 'is_color',
              'is_standard', 'C', 'kernel', 'ensemble_learning',
              'n_estimators']

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(self.request, 'algorithm/model_form.html',
                      {'form': form,
                       'step': 6,
                       'title': step_dict[6]['name'],
                       'remark': mark_safe('<button>1233</button>'),
                       })

    def get_success_url(self):
        return reverse_lazy('alogrithm:model_list')

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
            return render(self.request, 'algorithm/model_form.html',
                          {'form': form,
                           'step': 6,
                           'title': step_dict[6]['name'],
                           'remark': mark_safe('<button>1233</button>'),
                           'message': "The model_name is created!"
                           })

        form.instance.user_id = user_id

        # save the model in local
        svm_model = SVC(C=float(form.data['C']), kernel=form.data['kernel'], probability=True)

        if form.data['ensemble_learning'] == 'BaggingClassifier':
            svm_model = BaggingClassifier(base_estimator=svm_model, n_estimators=int(form.data['n_estimators']))
        elif form.data['ensemble_learning'] == 'AdaBoostClassifier':
            svm_model = AdaBoostClassifier(base_estimator=svm_model, n_estimators=int(form.data['n_estimators']))

        the_dir = path.join(settings.MEDIA_ROOT, 'upload_models', str(user_id))
        if not path.exists(the_dir):
            mkdir(the_dir)
        filename = form.data['model_name'] + '.model'
        the_path = path.join(the_dir, filename)
        with open(the_path, 'wb') as model_f:
            joblib.dump(svm_model, model_f)

        return super().form_valid(form)


class TrainSVMModelView(FormView):
    form_class = TrainLogForm

    def get_form(self, form_class=None):
        form_class = self.get_form_class()
        return form_class(self.request.user.id, **self.get_form_kwargs())

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(request, 'algorithm/train_svm_model.html',
                      {'form': form})

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form, **kwargs)
        else:
            return self.form_invalid(form, **kwargs)

    def form_invalid(self, form, **kwargs):
        return render(self.request, 'algorithm/train_svm_model.html',
                      {'form': form, 'message': form.errors
                       })

    def form_valid(self, form, **kwargs):
        user_id = self.request.user.id
        model_name = eval(form.data['model_name'])['model_name']
        train_category_positive_dict = eval(form.data['train_category_positive'])
        train_category_negative_dict = eval(form.data['train_category_negative'])

        train_category_positive = train_category_positive_dict['category']
        positive_num = train_category_positive_dict['num_category']
        train_category_negative = train_category_negative_dict['category']
        negative_num = train_category_negative_dict['num_category']
        validation_size = float(form.data['validation_size'])

        # train the model
        manager = mp.Manager()
        return_dict = manager.dict()
        proc = mp.Process(target=execute_train_model, args=(
            user_id, model_name, train_category_positive, train_category_negative, validation_size, return_dict
        ))
        proc.daemon = True
        proc.start()
        proc.join()

        train_log = ModelTrainLog(user_id=user_id, model_id=return_dict['model_id'],
                                  train_category_positive=train_category_positive,
                                  positive_num=positive_num,
                                  train_category_negative=train_category_negative,
                                  negative_num=negative_num,
                                  validation_size=validation_size,
                                  accuracy_score=return_dict['accuracy_score'] if validation_size != 0 else 0)
        train_log.save()

        form = self.get_form()
        # TODO：format the result in page
        return render(self.request, 'algorithm/train_svm_model.html',
                      {'form': form,
                       'accuracy_score': return_dict['accuracy_score'],
                       'classification_report': mark_safe(return_dict['classification_report']),
                       'confusion_matrix': mark_safe(return_dict['confusion_matrix'])})


class ModelListView(ListView):
    context_object_name = 'model_list'

    template_name = 'algorithm/model_list.html'

    def get_queryset(self):
        return SVMModel.objects.all(). \
            filter(user_id=self.request.user.id).order_by('recently_accuracy_score', '-update_time')


class ModelDetailView(View):

    def get(self, request, *args, **kwargs):
        pk = self.kwargs['pk']
        the_model = SVMModel.objects.all().filter(user_id=self.request.user.id, id=pk).values()[0]

        train_log = ModelTrainLog.objects.all().filter(model_id=the_model['id'])
        return render(request, 'algorithm/model_detail.html',
                      {
                          'model': the_model,
                          'train_log': train_log
                      })


# class ModelDetailView(DetailView):
#
#     def get_object(self, queryset=None):
#         pk = self.kwargs.get(self.pk_url_kwarg)
#         the_model = SVMModel.objects.filter(user_id=self.request.user.id, id=pk).get()
#         train_log = ModelTrainLog.objects.filter(model_id=the_model.id)
#         model_info = {
#             'model': the_model,
#             'train_log': train_log
#         }
#         return model_info
#
#     def get(self, request, *args, **kwargs):
#         model_info = self.get_object()
#         return render(request, 'algorithm/model_detail.html', {'model_info': model_info})
#

class ModelDeleteView(DeleteView):
    model = SVMModel
    success_url = reverse_lazy('alogrithm:model_list')

    def get(self, request, *args, **kwargs):
        return self.delete(request, *args, **kwargs)

    def get_queryset(self):
        return self.model.objects.filter(user_id=self.request.user.id)

    def delete(self, request, *args, **kwargs):
        user_id = self.request.user.id
        queryset = self.get_object()
        model_name = queryset.model_name

        model_path = saved_model_path(user_id, model_name)
        os.remove(model_path)

        response = super(ModelDeleteView, self).delete(request, *args, **kwargs)

        return response


class CatIdentificationView(FormView):
    form_class = CatIdentificationForm

    def get_form(self, form_class=None):
        form_class = self.get_form_class()
        return form_class(self.request.user.id, **self.get_form_kwargs())

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(self.request, 'algorithm/cat_identification.html',
                      {'form': form})

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        user_id = self.request.user.id
        model_name = eval(form.data['model_name'])['model_name']

        model_db = SVMModel.objects.get(user_id=user_id, model_name=model_name)
        if model_db.train_num == 0:
            cat_identification_form = CatIdentificationForm(request.user.id)
            return render(request, 'algorithm/cat_identification.html',
                          {'form': form,
                           'message': 'The trained model could predict, please train it'})
        else:
            files = request.FILES.getlist('file')
            show_probility = request.POST.get('show_probility')

            # save the files
            pre_pic_root = pre_pic_root_path(user_id)
            if path.exists(pre_pic_root):
                shutil.rmtree(pre_pic_root)
            mkdir(pre_pic_root)

            for f in files:
                pic_name = f.name
                with open(os.path.join(pre_pic_root, pic_name), 'wb+') as destination:
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

            return render(request, 'algorithm/cat_identification.html',
                          {'form': form,
                           'result': return_dict
                           })
