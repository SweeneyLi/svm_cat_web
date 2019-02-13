from django.db import models
from datetime import date


import os

from user.models import UserProfile


def user_directory_path(instance, filename):
    return os.path.join('upload_images', str(instance.user_id), instance.category, filename)


class Picture(models.Model):
    user_id = models.IntegerField('user_id')
    pic_name = models.CharField('pic_name', max_length=255, default='')
    path = models.ImageField("path", upload_to=user_directory_path, blank=False)
    category = models.CharField("category", max_length=100, blank=False, default='default')
    pic_size = models.CharField('pic_size', max_length=20, default='')
    upload_date = models.DateField(default=date.today)

    def __str__(self):
        return self.pic_name


# class UploadCategory(models.Model):
#     user_id = models.IntegerField('user_id')
#     cate_id = models.IntegerField('cate_id')
#
#     cate_num = models.IntegerField('cate_num')
#
#     update_time = models.DateTimeField('update_time', auto_now_add=True)
#     create_time = models.DateTimeField('create_time', auto_now=True)
#
#
# class UploadPicture(models.Model):
#     cate_name = models.CharField('cate_name', max_length=100, default='')
#
#     pic_name = models.CharField('pic_name', max_length=100, default='')
#     path = models.ImageField("path", upload_to=user_directory_path, blank=False)
#     pic_size = models.CharField('pic_size', max_length=20, default='')
#
#     upload_date = models.DateField(default=date.today)