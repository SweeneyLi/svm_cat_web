from django.urls import path, re_path
from . import views

# namespace
app_name = 'alogrithm'

urlpatterns = [
    path('choose_pic_category', views.choose_pic_category, name='choose_pic_category'),
    path('hog_pic/', views.hog_pic, name='hog_pic'),
    path('contrast_algorithm/', views.contrast_algorithm, name='contrast_algorithm'),
    path('create_model/', views.create_model, name='create_model'),

    # path('train_model/', views.train_model, name='train_model'),
]
