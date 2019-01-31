from django.urls import path, re_path
from . import views

# namespace
app_name = 'alogrithm'

urlpatterns = [
    path('choose_pic_category', views.choose_pic_category, name='choose_pic_category'),
    path('hog_pic/', views.hog_pic, name='hog_pic'),
    # path('pic_pro_visualization/', views.pic_pro_visualization, name='pic_pro_visualization'),
]
