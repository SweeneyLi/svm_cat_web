from django.db import models
from datetime import date


import os

from user.models import UserProfile


def user_directory_path(instance, filename):
    return os.path.join('upload_images', str(instance.user_id), instance.category, filename)


# Create your models here.
class Picture(models.Model):
    user_id = models.IntegerField('user_id')
    pic_name = models.CharField('pic_name', max_length=255, default='')
    path = models.ImageField("path", upload_to=user_directory_path, blank=False, default='')
    category = models.CharField("category", max_length=100, blank=False, default='default')
    pic_size = models.CharField('pic_size', max_length=20, default='')
    upload_date = models.DateField(default=date.today)

    def __str__(self):
        # // TODO: Chinese
        return self.pic_name

# # 对于使用Django自带的通用视图非常重要
#     def get_absolute_url(self):
#         return reverse('picture:pic_detail', args=[str(self.id)])
#
