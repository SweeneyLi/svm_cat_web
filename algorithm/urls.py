from django.urls import path, re_path
from . import views

# namespace
app_name = 'alogrithm'

urlpatterns = [

    re_path(r'^step/(?P<pk>\d+)/$', views.Step, name='step'),

    path('prepare_data/', views.PrepareDataView.as_view(), name='prepare_data'),
    path('hog_pic/', views.HOGPicView.as_view(), name='hog_pic'),
    path('eval_alg/', views.EvaluateAlgorithmView.as_view(), name='eval_alg'),
    path('adjust_svm/', views.AdjustSVMView.as_view(), name='adjust_svm'),
    path('adjust_ensemble_learning/', views.AdjustEnsembleLearningView.as_view(), name='adjust_ensemble_learning'),

    path('create_svm_model/', views.ModelCreateView.as_view(), name='create_svm_model'),
    path('train_svm_model/', views.TrainSVMModelView.as_view(), name='train_svm_model'),
    path('model_list/', views.ModelListView.as_view(), name='model_list'),
    re_path(r'^model_detail/(?P<pk>\d+)/$', views.ModelDetailView.as_view(), name='model_detail'),
    re_path(r'^model_delete/(?P<pk>\w+)/$', views.ModelDeleteView.as_view(), name='model_delete'),

    path('cat_identification/', views.CatIdentificationView.as_view(), name='cat_identification'),

    path('download_predict/', views.download_predict, name='download_predict'),
]
