from django.urls import path, re_path
from . import views

# namespace
app_name = 'alogrithm'

urlpatterns = [
    path('prepare_data', views.PrepareDataView.as_view(), name='prepare_data'),
    path('hog_pic/', views.HOGPicView.as_view(), name='hog_pic'),
    path('eval_alg/', views.EvaluateAlgorithmView.as_view(), name='eval_alg'),
    path('adjust_svm/', views.AdjustSVMView.as_view(), name='adjust_svm'),
    path('adjust_ensemble_learning/', views.AdjustEnsembleLearningView.as_view(), name='adjust_ensemble_learning'),

    path('create_svm_model/', views.ModelCreateView.as_view(), name='create_svm_model'),
    path('train_svm_model/', views.TrainSVMModelView.as_view(), name='train_svm_model'),
    path('model_list/', views.ModelListView.as_view(), name='model_list'),
    re_path(r'^model_detail/(?P<pk>\d+)/$', views.ModelDetail.as_view(), name='model_detail'),

    path('cat_identification/', views.cat_identification, name='cat_identification'),
]
