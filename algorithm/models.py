from django.db import models


class SVMModel(models.Model):
    model_name = models.CharField('model_name', max_length=50)
    create_time = models.DateTimeField('create_time', auto_now=True)
    update_time = models.DateTimeField('update_time', auto_now_add=True)
    train_num = models.IntegerField('train_num', default=0)
    accuracy_score = models.FloatField('accuracy_score', max_length=100, default=0)
    C = models.FloatField('C', max_length=50)
    kernel = models.CharField('kernel', max_length=50)
    pic_size = models.CharField('pic_size', max_length=50)
    orientations = models.CharField('orientations', max_length=50)
    pixels_per_cell = models.CharField('pixels_per_cell', max_length=50)
    cells_per_block = models.CharField('cells_per_block', max_length=50)
    is_color = models.BooleanField('is_color ')
    is_standard = models.BooleanField('is_standard')

