from os import path
from django.conf import settings
from django.urls import reverse_lazy

# params in svm
num_folds = 10
# seed = 7
seed = None
scoring = 'accuracy'


step_dict = {
    1: {
        'name': 'Prepare Data',
        'url': reverse_lazy('alogrithm:prepare_data')
    },
    2: {
        'name': 'Hog Picture',
        'url': reverse_lazy('alogrithm:hog_pic')
    },
    3: {
        'name': 'Evaluate algorithm',
        'url': reverse_lazy('alogrithm:eval_alg')
    },
    4: {
        'name': 'Adjust SVM',
        'url': reverse_lazy('alogrithm:adjust_svm')
    },
    5: {
        'name': 'Adjust Ensemble Learning',
        'url': reverse_lazy('alogrithm:adjust_ensemble_learning')
    },
    6: {
        'name': 'Create SVM Model',
        'url': reverse_lazy('alogrithm:create_svm_model')
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


