from os import path
from django.conf import settings


# save the input pic
def saved_pic_path(user_id):
    return path.join(settings.MEDIA_ROOT, 'algorithm', 'hog_picture',
                     user_id + '_' + 'hog_test_pic.jpg')


def hog_pic_path(user_id):
    return path.join('/media', 'algorithm', 'hog_picture',
                     user_id + '_hog_picture.png')


# params in svm
num_folds = 10
seed = 7
scoring = 'accuracy'
