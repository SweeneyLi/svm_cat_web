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
            'Prepare Data': 'You need to prepare the data to adjust parameter.',
            'Test Pic': 'The picture you will use to find hog parameter. '
                        'Please upload the representative and classial picture in you category. ',
            'Test category positive': 'The category which you want to identificate finally.',
            'Test category negative': 'The category which you want to discriminate to positive category finally.',
            'Validation Size': 'Later we will need to test the algorithm.  '
                               'This size is used to split the train and test.'
                               ' The default value is 0.2. '
                               '(Which represent 80% train and 20% test of you uploaded two category)',
        },
    },
    'hogPic': {
        'step': 2,
        'title': 'Hog Picture',
        'url': reverse_lazy('alogrithm:hog_pic'),
        'help_text': {
            'Hog Picture': 'You can adjust the parameter of hog which the function of changing picture to feature vectors.',
            'The hog': 'Histogram of Oriented Gradients. '
                       'The histogram of oriented gradients (HOG) is a feature descriptor used in computer vision and image processing for the purpose of object detection. ',
            'Picture Size': 'The size of picture.',
            'orientations': 'The nbins option specifies the number of bins to be included in the histogram and then split at the best point. These split points are evaluated at the boundaries of each of these bins.',
            'Pixels per cell:': 'The size of cell.',
            'Cells per block': 'Decide the num of cells in each block.',
            'Is color': 'Whether use the colorful picture.',
        },
    },
    'evalAlg': {
        'step': 3,
        'title': 'Evaluate algorithm',
        'url': reverse_lazy('alogrithm:eval_alg'),
        'help_text': {
            'evalAlg': 'You can evaluate the SVM algorithm to other algorithm(With the default parameter) and decided to statndard the vector.',
            'Is standard': 'Whether to standardize the feature of picture. '
                           'Recommend to standard.',
            'algorithm_list': 'Tha algorithm you want to contrast with SVM',
        },
    },
    'adjustSVM': {
        'step': 4,
        'title': 'Adjust SVM',
        'url': reverse_lazy('alogrithm:adjust_svm'),
        'help_text': {
            'adjustSVM': 'You can adjust the parameter in SVM Model',
            'C': 'The C parameter tells the SVM optimization how much you want to avoid misclassifying each training example.  '
                 'For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly.  '
                 'Conversely, a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane,  '
                 'even if that hyperplane misclassifies more points.',
            'kernel': 'The kernel methods are a class of algorithms for pattern analysis. '
                      'The general task of pattern analysis is to find and study general types of relations (for example clusters, rankings, principal components, correlations, classifications) in datasets. In its simplest form, the kernel trick means transforming data into another dimension that has a clear dividing margin between classes of data.',
        },
    },
    'adjustEnsembleLearning': {
        'step': 5,
        'title': 'Adjust Ensemble Learning',
        'url': reverse_lazy('alogrithm:adjust_ensemble_learning'),
        'help_text': {
            'adjustEnsembleLearning': 'You can decide to use the ensemble learning',
            'C': 'As same as previous page',
            'Kernel': 'As same as previous page',
            'Ensemble learning': 'Ensemble methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone.',
            'N estimators': 'The num of estimators.'

        },
    },
    'createSVMModel': {
        'step': 6,
        'title': 'Create SVM Model',
        'url': reverse_lazy('alogrithm:create_svm_model'),
        'help_text': {
            'Create SVM Model': 'You can create model with previous adjusted parameter.',
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


def predict_json_path(user_id):
    return path.join(settings.MEDIA_ROOT, 'predict_jsons', str(user_id) + '.json')
