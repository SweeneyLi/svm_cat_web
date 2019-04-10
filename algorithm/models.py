from django.db import models
from django.contrib.auth.models import User


class SVMModel(models.Model):
    user_id = models.IntegerField('user_id',
                                  )
    model_name = models.CharField(verbose_name='模型名称', max_length=50)
    comment = models.TextField(verbose_name='备注', max_length=255, blank=True)

    train_num = models.IntegerField('train_num', default=0)
    update_time = models.DateTimeField('update_time', auto_now_add=True)
    recently_accuracy_score = models.FloatField('recently_accuracy_score', max_length=100, default=0)

    C = models.FloatField(verbose_name='惩罚系数', max_length=50)
    kernel_choice = [('sigmoid', 'sigmoid'), ('rbf', 'rbf'), ('poly', 'poly'), ('linear', 'linear')]
    kernel = models.CharField(verbose_name='核函数', choices=kernel_choice, max_length=50)

    is_standard = models.BooleanField(verbose_name='是否标准化')
    is_color = models.BooleanField(verbose_name='是否带颜色')
    pic_size = models.CharField(verbose_name='图片大小', max_length=50)
    orientations = models.IntegerField(verbose_name='bin的个数')
    pixels_per_cell = models.CharField(verbose_name='每个cell的像素数', max_length=50)
    cells_per_block = models.CharField(verbose_name='每个BLOCK内cell大小', max_length=50)

    create_time = models.DateTimeField('create_time', auto_now=True)

    ensemble_learning_choice = [('BaggingClassifier', 'BaggingClassifier'),
                                ('AdaBoostClassifier', 'AdaBoostClassifier'),
                                ('None', 'None')]
    ensemble_learning = models.CharField(verbose_name='集成算法', choices=ensemble_learning_choice, max_length=50,
                                         default='None')
    n_estimators = models.IntegerField(verbose_name='决策树个数', default=0)

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
