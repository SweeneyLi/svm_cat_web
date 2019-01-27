from django.urls import path
from . import views

app_name = 'system'
urlpatterns = [
    path('index/', views.index, name='main_view'),

]