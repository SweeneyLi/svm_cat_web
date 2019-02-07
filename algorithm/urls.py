from django.urls import path, re_path
from . import views

# namespace
app_name = 'alogrithm'

urlpatterns = [
    path('choose_pic_category', views.choose_pic_category, name='choose_pic_category'),
    path('hog_pic/', views.hog_pic, name='hog_pic'),
    path('contrast_algorithm/', views.contrast_algorithm, name='contrast_algorithm'),
    path('adjust_svm/', views.adjust_svm, name='adjust_svm'),
    path('create_svm_model/', views.ModelCreateView.as_view(), name='create_svm_model'),
    path('train_svm_model/', views.ModelTrainView.as_view(), name='train_svm_model'),
    # re_path(r'^train_svm_model/(?P<model_name>\w+)/', views.ModelTrainView.as_view(), name='train_svm_model'),
    # path('train_model/', views.train_model, name='train_model'),
]
