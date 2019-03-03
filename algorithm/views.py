from django.shortcuts import render, reverse, redirect
from django.views.generic import CreateView, ListView, DetailView, FormView, DeleteView, View
from django.utils.safestring import mark_safe

from system.url_conf import url_dict
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
def step(request, pk):
    pk = int(pk)
    user_id = str(request.user.id)
    if pk < 1:
        return redirect(reverse_lazy('picture:pic_upload'))
    elif pk > 7:
        return redirect(reverse_lazy('alogrithm:train_svm_model'))
    else:

        # judege the category
        with open(settings.ALGORITHM_JSON_PATH, "r") as load_f:
            algorithm_info = json.load(load_f)

        if pk > 1 and algorithm_info[user_id]['data_para'] == {}:
            # form = PrepareDataForm(user_id=pk)
            message = 'You should set the test category positive and <br> ' \
                      'test category positive in the first step firstly!'
            return redirect(reverse_lazy('alogrithm:prepare_data') + "?message=" + message)
        else:
            for _, value in step_info.items():
                if value['step'] == pk:
                    return redirect(value['url'])


# template
# class template(FormView):
#     form_class = PrepareDataForm
#
#     def get_form(self, form_class=None):
#         pass
#
#     def get(self, request, *args, **kwargs):
#         form = None
#
#         return render(request, 'algorithm/model_form.html',
#                       {'form': form,
#                        'step': 2,
#                        'title': step_dict[2]['name'],
#                        'remark': mark_safe('<button>1233</button>'),
#                        'picture': None
#                        })
#
#     def post(self, request, *args, **kwargs):
#         form = self.get_form()
#         if form.is_valid():
#             return self.form_valid(form, **kwargs)
#         else:
#             return self.form_invalid(form, **kwargs)
#
#     def form_invalid(self, form, **kwargs):
#         return render(self.request, 'algorithm/choose_pic_category.html',
#                       {'form': form, 'message': form.errors
#                        })
#
#     def form_valid(self, form, **kwargs):
#         pass
#

class PrepareDataView(FormView):
    form_class = PrepareDataForm
    view_name = 'prepareData'

    def get_form(self, form_class=None):
        form_class = self.get_form_class()
        return form_class(self.request.user.id, **self.get_form_kwargs())

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        message = request.GET.get('message', None)
        return render(request, 'algorithm/model_form.html',
                      {'form': form,
                       'url_info': step_info[self.view_name],
                       'message': message
                       }
                      )

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form, **kwargs)
        else:
            return self.form_invalid(form, **kwargs)

    def form_invalid(self, form, **kwargs):
        return render(self.request, 'algorithm/model_form.html',
                      {'form': form,
                       'url_info': step_info[self.view_name],
                       }
                      )

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
    view_name = 'hogPic'

    def get(self, request, *args, **kwargs):

        user_id = str(request.user.id)
        if not os.path.exists(saved_pic_path(user_id)):
            message = "Please set the test picture in the first step firstly!"
            return redirect(reverse_lazy('alogrithm:prepare_data') + "?message=" + message)
        else:
            form = self.get_form()
            return render(request, 'algorithm/model_form.html',
                          {'form': form,
                           'url_info': step_info[self.view_name],
                           }
                          )

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form, **kwargs)
        else:
            return self.form_invalid(form, **kwargs)

    def form_invalid(self, form, **kwargs):

        return render(self.request, 'algorithm/model_form.html',
                      {'form': form,
                       'url_info': step_info[self.view_name],
                       }
                      )

    def form_valid(self, form, **kwargs):
        to_tuple = lambda str: tuple((int(str.split(',')[0]), int(str.split(',')[1])))

        pic_size = to_tuple(form.data['pic_size'])
        orientations = int(form.data['orientations'])
        pixels_per_cell = to_tuple(form.data['pixels_per_cell'])
        cells_per_block = to_tuple(form.data['cells_per_block'])
        is_color = True if 'is_color' in form.data else False
        user_id = str(self.request.user.id)

        proc = mp.Process(target=execute_hog_pic,
                          args=(pic_size, orientations, pixels_per_cell, cells_per_block, is_color, user_id))
        proc.daemon = True
        proc.start()
        proc.join()

        # get the saved png to show in page
        hog_pic = hog_pic_path(user_id)

        return render(self.request, 'algorithm/model_form.html',
                      {'form': form,
                       'url_info': step_info[self.view_name],
                       'picture': hog_pic,
                       }
                      )


class EvaluateAlgorithmView(FormView):
    form_class = EvaluateAlgoritmForm
    view_name = 'evalAlg'

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(request, 'algorithm/model_form.html',
                      {'form': form,
                       'url_info': step_info[self.view_name],
                       }
                      )

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
                       'url_info': step_info[self.view_name],
                       'picture': eval_pic,
                       }
                      )


class AdjustSVMView(FormView):
    form_class = SVMParameterForm
    view_name = 'adjustSVM'

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(request, 'algorithm/model_form.html',
                      {'form': form,
                       'url_info': step_info[self.view_name],
                       }
                      )

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form, **kwargs)
        else:
            return self.form_invalid(form, **kwargs)

    def form_invalid(self, form, **kwargs):
        return render(self.request, 'algorithm/model_form.html',
                      {'form': form,
                       'url_info': step_info[self.view_name],
                       }
                      )

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
                       'url_info': step_info[self.view_name],
                       'results': return_dict,
                       }
                      )


class AdjustEnsembleLearningView(FormView):
    form_class = EnsembleParamsForm
    view_name = 'adjustEnsembleLearning'

    def get_form(self, form_class=None):
        form_class = self.get_form_class()
        return form_class(self.request.user.id, **self.get_form_kwargs())

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(request, 'algorithm/model_form.html',
                      {'form': form,
                       'url_info': step_info[self.view_name],
                       }
                      )

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form, **kwargs)
        else:
            return self.form_invalid(form, **kwargs)

    def form_invalid(self, form, **kwargs):
        return render(self.request, 'algorithm/model_form.html',
                      {'form': form,
                       'url_info': step_info[self.view_name],
                       }
                      )

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
                       'url_info': step_info[self.view_name],
                       'results': return_dict,
                       }
                      )


class ModelCreateView(CreateView):
    template_name = 'algorithm/model_form.html'
    model = SVMModel
    fields = ['model_name', 'comment', 'pic_size', 'orientations',
              'pixels_per_cell', 'cells_per_block', 'is_color',
              'is_standard', 'C', 'kernel', 'ensemble_learning',
              'n_estimators']
    view_name = 'createSVMModel'

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

        model_best_params = algorithm_info[user_id]['model_para'].get('best_params',
                                                                      {'C': 2.0, 'kernel': 'sigmoid'})
        kwargs['initial']['C'] = model_best_params['C']
        kwargs['initial']['kernel'] = model_best_params['kernel']

        kwargs['initial']['ensemble_learning'] = algorithm_info[user_id]['ensemble_para'].get('ensemble_learning',
                                                                                              'None')
        kwargs['initial']['n_estimators'] = algorithm_info[user_id]['ensemble_para'].get('n_estimators', 0)

        return kwargs

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(request, 'algorithm/model_form.html',
                      {'form': form,
                       'url_info': step_info[self.view_name],
                       }
                      )

    def form_valid(self, form):
        # TODO: decide to delete pkl
        user_id = self.request.user.id

        # judge the same model_name
        if SVMModel.objects.filter(user_id=user_id, model_name=form.data['model_name']).exists():
            return render(self.request, 'algorithm/model_form.html',
                          {'form': form,
                           'url_info': step_info[self.view_name],
                           'message': "The model_name is created!"
                           }
                          )

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

    def get_success_url(self):
        return reverse_lazy('alogrithm:model_list')


class TrainSVMModelView(FormView):
    form_class = TrainLogForm
    view_name = 'trainSVMModel'

    def get_form(self, form_class=None, reset=False):
        form_class = self.get_form_class()
        if reset:
            return form_class(self.request.user.id)
        else:
            return form_class(self.request.user.id, **self.get_form_kwargs())

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(request, 'system/common_form.html',
                      {'form': form,
                       'url_info': url_dict[self.view_name],
                       })

    def post(self, request, *args, **kwargs):
        form = self.get_form()

        if form.is_valid():
            return self.form_valid(form, **kwargs)
        else:
            return self.form_invalid(form, **kwargs)

    def form_invalid(self, form, **kwargs):
        return render(self.request, 'system/common_form.html',
                      {'form': form,
                       'url_info': url_dict[self.view_name],
                       })

    def form_valid(self, form, **kwargs):
        user_id = self.request.user.id
        model_name = eval(self.request.POST['model_name'])['model_name']
        train_category_positive_dict = eval(self.request.POST['train_category_positive'])
        train_category_negative_dict = eval(self.request.POST['train_category_negative'])

        train_category_positive = train_category_positive_dict['category']
        positive_num = train_category_positive_dict['num_category']
        train_category_negative = train_category_negative_dict['category']
        negative_num = train_category_negative_dict['num_category']
        validation_size = float(self.request.POST['validation_size'])

        # train the model
        manager = mp.Manager()
        return_dict = manager.dict()
        proc = mp.Process(target=execute_train_model, args=(
            user_id, model_name, train_category_positive, train_category_negative, validation_size, return_dict
        ))
        proc.daemon = True
        proc.start()
        proc.join()

        if not return_dict.get('Errors'):
            train_log = ModelTrainLog(user_id=user_id, model_id=return_dict['model_id'],
                                      train_category_positive=train_category_positive,
                                      positive_num=positive_num,
                                      train_category_negative=train_category_negative,
                                      negative_num=negative_num,
                                      validation_size=validation_size,
                                      accuracy_score=return_dict['accuracy_score'] if validation_size != 0 else 0)
            train_log.save()
            del return_dict['model_id']
        # TODO:error problem
        form = self.get_form(reset=True)
        # TODO：format the result in page
        return render(self.request, 'system/common_form.html',
                      {'form': form,
                       'url_info': url_dict[self.view_name],
                       'result': return_dict}
                      )


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
    view_name = 'catIdentification'

    def get_form(self, form_class=None, reset=False):
        form_class = self.get_form_class()
        if reset:
            return form_class(self.request.user.id)
        else:
            return form_class(self.request.user.id, **self.get_form_kwargs())

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(self.request, 'system/common_form.html',
                      {'form': form,
                       'url_info': url_dict[self.view_name]})

    def post(self, request, *args, **kwargs):
        form = self.get_form(reset=True)
        user_id = self.request.user.id
        model_name = eval(request.POST['model_name'])['model_name']

        model_db = SVMModel.objects.get(user_id=user_id, model_name=model_name)
        if model_db.train_num == 0:
            return render(request, 'system/common_form.html',
                          {'form': form,
                           'url_info': url_dict[self.view_name],
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

            return render(request, 'system/common_form.html',
                          {'form': form,
                           'result': return_dict,
                           'url_info': url_dict[self.view_name]
                           })
