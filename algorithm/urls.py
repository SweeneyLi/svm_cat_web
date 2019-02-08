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
    path('train_svm_model/', views.train_svm_model, name='train_svm_model'),
    path('model_list/', views.ModelListView.as_view(), name='model_list'),
    re_path(r'^model_detail/(?P<pk>\d+)/$', views.ModelDetail.as_view(), name='model_detail'),
    path('adjust_ensemble_learning/', views.adjust_ensemble_learning, name='adjust_ensemble_learning'),
    path('cat_identification/', views.cat_identification, name='cat_identification'),
]
