from django.db import models
from datetime import date
from django.urls import reverse
from django.conf import settings
import os
from user.models import UserProfile


def user_directory_path(instance, filename):

    return os.path.join(instance.user.id, instance.category, filename)


# Create your models here.
class Picture(models.Model):

    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='picture', default='')
    pic_name = models.CharField("图片名", max_length=100, blank=True, default='')
    category = models.CharField("类别", max_length=100, blank=True, default='default')
    image = models.ImageField("图片", upload_to=user_directory_path, blank=True)
    upload_date = models.DateField(default=date.today)

    def __str__(self):
        return self.pic_name

# 对于使用Django自带的通用视图非常重要
    def get_absolute_url(self):
        return reverse('picture:pic_detail', args=[str(self.id)])

