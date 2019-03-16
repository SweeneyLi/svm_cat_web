from django.urls import path
from . import views

app_name = 'system'
urlpatterns = [
    path('', views.index, name='default'),
    path('index/', views.index, name='index'),
]
