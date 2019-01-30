from django.urls import path, re_path
from . import views

# namespace
app_name = 'alogrithm'

urlpatterns = [
    path('pic_processing/', views.pic_processing, name='pic_processing'),
    path('pic_pro_visualization/', views.pic_pro_visualization, name='pic_pro_visualization'),
]
