from django import template
from django.conf import settings
from random import randint
import os
import re

register = template.Library()


@register.filter()
def num_to_English(num):
    num_English_dict = {
        1: 'First',
        2: 'Second',
        3: 'Third',
        4: 'Forth',
        5: 'Fifrh',
        6: 'Sixth',
        7: 'Seventh'
    }
    return num_English_dict[int(num)]


@register.filter()
def field_type(field):
    return re.findall(r'"\w+"', str(field))[0].strip('"')


@register.filter()
def label_with_classes(value, arg):
    return value.label_tag(attrs={'class': arg})


@register.filter()
def widget_with_classes(value, arg):
    return value.as_widget(attrs={'class': arg})


@register.filter()
def random_pic(path):
    if path:
        return path
    else:
        root_path = os.path.join((settings.STATICFILES_DIRS)[0], 'img', 'LXH', 'gif')
        len_num = len([name for name in os.listdir(root_path)])
        return os.path.join(settings.STATIC_URL, 'img', 'LXH', 'gif', 'gif_' + str(randint(1, len_num)) + '.gif')
