from django.db import models
from datetime import date
from django.urls import reverse

import os
from django.contrib.auth.models import User


def user_directory_path(instance, filename):

    return os.path.join(instance.user.id, instance.category, filename)


# Create your models here.
class Picture(models.Model):

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='picture')
    pic_name = models.CharField("图片名", max_length=100, blank=True, default='')
    category = models.CharField("类别", max_length=100, blank=True, default='')
    image = models.ImageField("图片", upload_to=user_directory_path, blank=True)
    upload_date = models.DateField(default=date.today)

    def __str__(self):
        return self.pic_name

# 对于使用Django自带的通用视图非常重要
    def get_absolute_url(self):
        return reverse('picture:pic_detail', args=[str(self.id)])

