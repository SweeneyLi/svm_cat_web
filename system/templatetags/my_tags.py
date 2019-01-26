from django import template
from django.utils.safestring import mark_safe

register = template.Library()


# @register.simple_tag
# def add(a,b):
#     return a+b
#
#
# @register.filter(name='get_pic_name')
# def get_pic_name(path):
#     return path.split('/')[-1]
