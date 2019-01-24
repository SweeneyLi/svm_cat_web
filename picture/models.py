from django.db import models
from datetime import date
from django.urls import reverse
import os
from user.models import UserProfile


def user_directory_path(instance, filename):
    if instance.pic_name.lower().split('.')[-1] in ['jpg', 'png', 'jpeg']:
        name = instance.pic_name
    else:
        name = instance.pic_name + '.' + filename.split('.')[-1]

    return os.path.join(str(instance.user_id), instance.category, name)


# Create your models here.
class Picture(models.Model):

    user_id = models.IntegerField('user_id', default=1)
    pic_name = models.CharField("图片名", max_length=100, blank=False, default='default')
    category = models.CharField("类别", max_length=100, blank=False, default='default')
    image = models.ImageField("图片", upload_to=user_directory_path, blank=False)
    upload_date = models.DateField(default=date.today)

    def __str__(self):
        return self.pic_name

# 对于使用Django自带的通用视图非常重要
    def get_absolute_url(self):
        return reverse('picture:pic_detail', args=[str(self.id)])

