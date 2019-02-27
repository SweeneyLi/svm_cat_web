from os import path
from django.conf import settings
from django.urls import reverse_lazy

# params in svm
num_folds = 10
# seed = 7
seed = None
scoring = 'accuracy'

step_info = {
    'prepareData': {
        'step': 1,
        'title': 'Prepare Data',
        'url': reverse_lazy('alogrithm:prepare_data'),
        'help_text': {
            'Pic': '',
            '2': '',
            '3': '',
            '4': '',
        },
    },
    'hogPic': {
        'step': 2,
        'title': 'Hog Picture',
        'url': reverse_lazy('alogrithm:hog_pic'),
        'help_text': {
            '1': '',
            '2': '',
            '3': '',
            '4': '',
        },
    },
    'evalAlg': {
        'step': 3,
        'title': 'Evaluate algorithm',
        'url': reverse_lazy('alogrithm:eval_alg'),
        'help_text': {
            '1': '',
            '2': '',
            '3': '',
            '4': '',
        },
    },
    'adjustSVM': {
        'step': 4,
        'title': 'Adjust SVM',
        'url': reverse_lazy('alogrithm:adjust_svm'),
        'help_text': {
            '1': '',
            '2': '',
            '3': '',
            '4': '',
        },
    },
    'adjustEnsembleLearning': {
        'step': 5,
        'title': 'Adjust Ensemble Learning',
        'url': reverse_lazy('alogrithm:adjust_ensemble_learning'),
        'help_text': {
            '1': '',
            '2': '',
            '3': '',
            '4': '',
        },
    },
    'createSVMModel': {
        'step': 6,
        'title': 'Create SVM Model',
        'url': reverse_lazy('alogrithm:create_svm_model'),
        'help_text': {
            '1': '',
            '2': '',
            '3': '',
            '4': '',
        },
    },

}


# save the input pic
def saved_pic_path(user_id):
    return path.join(settings.MEDIA_ROOT, 'algorithm', 'hog_picture',
                     user_id + '_' + 'hog_test_pic.jpg')


def hog_pic_path(user_id, relative=True):
    if relative:
        return path.join('/media', 'algorithm', 'hog_picture',
                         user_id + '_hog_picture.png')
    else:
        return path.join(settings.MEDIA_ROOT, 'algorithm', 'hog_picture',
                         user_id + '_hog_picture.png')


def eval_pic_path(user_id, relative=True):
    if relative:
        return path.join('/media', 'algorithm', 'hog_picture',
                         user_id + '_evaluate_algorithm.png')
    else:
        return path.join(settings.MEDIA_ROOT, 'algorithm', 'hog_picture',
                         user_id + '_evaluate_algorithm.png')


def pre_pic_root_path(user_id):
    return path.join(settings.MEDIA_ROOT, 'predict_images',
                     str(user_id))


def saved_model_path(user_id, model_name):
    return path.join(settings.MEDIA_ROOT, 'upload_models', str(user_id), model_name + '.model')
