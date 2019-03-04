from django.db import models
from django.contrib.auth.models import User


class SVMModel(models.Model):
    user_id = models.IntegerField('user_id')
    model_name = models.CharField('model_name', max_length=50)
    comment = models.TextField('comment', max_length=255, blank=True)

    train_num = models.IntegerField('train_num', default=0)
    update_time = models.DateTimeField('update_time', auto_now_add=True)
    recently_accuracy_score = models.FloatField('recently_accuracy_score', max_length=100, default=0)

    C = models.FloatField('C', max_length=50)
    kernel_choice = [('sigmoid', 'sigmoid'), ('rbf', 'rbf'), ('poly', 'poly'), ('linear', 'linear')]
    kernel = models.CharField('kernel', choices=kernel_choice, max_length=50)

    is_standard = models.BooleanField('is_standard')
    is_color = models.BooleanField('is_color')
    pic_size = models.CharField('pic_size', max_length=50)
    orientations = models.IntegerField('orientations')
    pixels_per_cell = models.CharField('pixels_per_cell', max_length=50)
    cells_per_block = models.CharField('cells_per_block', max_length=50)

    create_time = models.DateTimeField('create_time', auto_now=True)

    ensemble_learning_choice = [('BaggingClassifier', 'BaggingClassifier'),
                                ('AdaBoostClassifier', 'AdaBoostClassifier'),
                                ('None', 'None')]
    ensemble_learning = models.CharField('ensemble_learning', choices=ensemble_learning_choice, max_length=50,
                                         default='None')
    n_estimators = models.IntegerField('n_estimators', default=0)

    def __str__(self):
        return self.model_name


class ModelTrainLog(models.Model):
    user_id = models.IntegerField('user_id')
    train_time = models.DateTimeField('train_time', auto_now_add=True)
    model_id = models.IntegerField('model_id')

    accuracy_score = models.FloatField('accuracy_score', max_length=100, default=0)
    train_category_positive = models.CharField('train_category_positive', max_length=100)
    positive_num = models.IntegerField('positive_num')
    train_category_negative = models.CharField('train_category_negative', max_length=100)
    negative_num = models.IntegerField('negative_num')
    validation_size = models.FloatField('validation_size')

