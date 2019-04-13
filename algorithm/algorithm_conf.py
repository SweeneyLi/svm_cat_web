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
        'title': '准备数据',  # 'Prepare Data',
        'url': reverse_lazy('alogrithm:prepare_data'),
        'help_text': {
            '准备数据': '您需要在此页面配置测试的数据信息.',
            '测试图片': '上传一张图片（正确的）用来调试出合适的图片处理参数。',
            '正确的图片类别': '请选择您想要识别的图片类别， 图片类别最少十张。',
            '错误的图片类别': '请选择您想要区分的图片类别， 图片类别最少十张。',
            '数据集比例': '此数值用于决定以上的两个图片类别中用于训练的图片比例。'
            # 'Prepare Data': 'You need to prepare the data to adjust parameter.',
            # 'Test Pic': 'The picture you will use to find hog parameter. '
            #             'Please upload the representative and classial picture in you category. ',
            # 'Test category positive': 'The category which you want to identificate finally.',
            # 'Test category negative': 'The category which you want to discriminate to positive category finally.',
            # 'Validation Size': 'Later we will need to test the algorithm.  '
            #                    'This size is used to split the train and test.'
            #                    ' The default value is 0.2. '
            #                    '(Which represent 80% train and 20% test of you uploaded two category)',
        },
    },
    'hogPic': {
        'step': 2,
        'title': '处理图片',  # 'Hog Picture',
        'url': reverse_lazy('alogrithm:hog_pic'),
        'help_text': {
            '处理图片': '您需要在此页面调试出符合数据集的图片处理参数，将图片转成合适维度的向量。',
            '图片大小': '图片的大小。',
            'bins个数': '指定bin的个数. 本系统实现的只有无符号方向.'
                      '''(根据反正切函数的到的角度范围是在-180°~ 180°之间, '''
                      '''无符号是指把 -180°~0°这个范围统一加上180°转换到0°~180°范围内. 有符号是指将-180°~180°转换到0°~360°范围内.)'''
                      '''也就是说把所有的方向都转换为0°~180°内, 然后按照指定的orientation数量划分bins. 比如你选定的orientation= 9, 则bin一共有9个, 每20°一个:'''
                      '''[0°~20°, 20°~40°, 40°~60° 60°~80° 80°~100°, 100°~120°, 120°~140°, 140°~160°, 160°~180°]''',
            '每个cell的像素数': '每个cell的像素数, 是一个tuple类型数据,例如(20,20).',
            '每个BLOCK内cell分布': '每个BLOCK内有多少个cell, tuple类型, 例如(2,2), 意思是将block均匀划分为2x2的块.',
            '是否选择颜色': '图片是否带颜色。',
            # 'Hog Picture': 'You can adjust the parameter of hog which the function of changing picture to feature vectors.',
            # 'The hog': 'Histogram of Oriented Gradients. '
            #            'The histogram of oriented gradients (HOG) is a feature descriptor used in computer vision and image processing for the purpose of object detection. ',
            # 'Picture Size': 'The size of picture.',
            # 'orientations': 'The nbins option specifies the number of bins to be included in the histogram and then split at the best point. These split points are evaluated at the boundaries of each of these bins.',
            # 'Pixels per cell:': 'The size of cell.',
            # 'Cells per block': 'Decide the num of cells in each block.',
            # 'Is color': 'Whether use the colorful picture.',
        },
    },
    'evalAlg': {
        'step': 3,
        'title': '集成算法',  # 'Evaluate algorithm',
        'url': reverse_lazy('alogrithm:eval_alg'),
        'help_text': {
            '评估算法': '在此页面可以将 SVM 与其他的算法进行对比， 以及决定是否数据标准化',
            '算法列表': '不同算法之间的对比（默认参数）。',
            '是否数据标准化': '将图片转成的向量标准化。',
            # 'evalAlg': 'You can evaluate the SVM algorithm to other algorithm(With the default parameter) and decided to statndard the vector.',
            # 'Is standard': 'Whether to standardize the feature of picture. '
            #                'Recommend to standard.',
            # 'algorithm_list': 'Tha algorithm you want to contrast with SVM',
        },
    },
    'adjustSVM': {
        'step': 4,
        'title': 'SVM调参',  # 'Adjust SVM',
        'url': reverse_lazy('alogrithm:adjust_svm'),
        'help_text': {
            'SVM调参': '在此页面调整 SVM 模型的参数。',
            'C': '惩罚系数C为调节优化方向中两个指标（间隔大小，分类准确度）偏好的权重',
            '核函数': 'kernel其实就是帮我们省去在高维空间里进行繁琐计算的“简便运算法”。',
            # 'adjustSVM': 'You can adjust the parameter in SVM Model',
            # 'C': 'The C parameter tells the SVM optimization how much you want to avoid misclassifying each training example.  '
            #      'For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly.  '
            #      'Conversely, a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane,  '
            #      'even if that hyperplane misclassifies more points.',
            # 'kernel': 'The kernel methods are a class of algorithms for pattern analysis. '
            #           'The general task of pattern analysis is to find and study general types of relations (for example clusters, rankings, principal components, correlations, classifications) in datasets. '
            #           'In its simplest form, the kernel trick means transforming data into another dimension that has a clear dividing margin between classes of data.',
        },
    },
    'adjustEnsembleLearning': {
        'step': 5,
        'title': '集成算法',  # 'Adjust Ensemble Learning',
        'url': reverse_lazy('alogrithm:adjust_ensemble_learning'),
        'help_text': {
            '集成学习算法': '您可以决定是否选择采用集成算法，在提高准确率的同时也会导致速度的降低。',
            'C': '惩罚系数C为调节优化方向中两个指标（间隔大小，分类准确度）偏好的权重',
            '核函数': 'kernel其实就是帮我们省去在高维空间里进行繁琐计算的“简便运算法”。',
            '决策树数量': '集成算法中的决策树的数量'
            # 'adjustEnsembleLearning': 'You can decide to use the ensemble learning',
            # 'C': 'As same as previous page',
            # 'Kernel': 'As same as previous page',
            # 'Ensemble learning': 'Ensemble methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone.',
            # 'N estimators': 'The num of estimators.'

        },
    },
    'createSVMModel': {
        'step': 6,
        'title': '建立模型',  # 'Create SVM Model',
        'url': reverse_lazy('alogrithm:create_svm_model'),
        'help_text': {
            '建立模型': '您可以根据之前五个步骤所选择的参数建立属于您的 SVM 模型。',
            # 'Create SVM Model': 'You can create model with previous adjusted parameter.',
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
