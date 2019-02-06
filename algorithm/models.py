from django.db import models
from django.contrib.auth.models import User


class SVMModel(models.Model):
    user_id = models.IntegerField('user_id')
    model_name = models.CharField('model_name', max_length=50)
    train_num = models.IntegerField('train_num', default=0)
    create_time = models.DateTimeField('create_time', auto_now=True)
    update_time = models.DateTimeField('update_time', auto_now_add=True)

    accuracy_score = models.FloatField('accuracy_score', max_length=100, default=0)
    C = models.FloatField('C', max_length=50)
    kernel = models.CharField('kernel', max_length=50)
    is_standard = models.BooleanField('is_standard')
    pic_size = models.CharField('pic_size', max_length=50)
    orientations = models.CharField('orientations', max_length=50)
    pixels_per_cell = models.CharField('pixels_per_cell', max_length=50)
    cells_per_block = models.CharField('cells_per_block', max_length=50)
    is_color = models.BooleanField('is_color ')

    def __str__(self):
        # // TODO: Chinese
        return self.model_name

# class ModelTrainLog(models.Model):
#     train_time = models.DateTimeField('train_time', auto_now_add=True)
#
