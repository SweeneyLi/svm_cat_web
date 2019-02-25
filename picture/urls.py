from django.urls import path, re_path
from . import views

# namespace
app_name = 'picture'

urlpatterns = [

    path('pic_list/', views.PicListView.as_view(), name='pic_list'),

    re_path(r'^upload/$', views.PicUploadView.as_view(), name='pic_upload'),

    re_path(r'^pic_detail/(?P<pk>\d+)/$', views.PicDetailView.as_view(), name='pic_detail'),

    re_path(r'^pic_delete/(?P<pk>\w+)/$', views.PicDeleteView.as_view(), name='pic_delete'),
]
