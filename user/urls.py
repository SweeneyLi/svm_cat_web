from django.urls import re_path, path
from . import views

app_name = 'user'
urlpatterns = [
    re_path(r'^register/$', views.register, name='register'),
    re_path(r'^login/$', views.login, name='login'),
    path('user/profile/', views.profile, name='profile'),
    path('user/profileUpdate/', views.ProfileUpdateView.as_view(), name='profile_update'),
    path('user/pwdchange/', views.pwd_change, name='pwd_change'),
    re_path(r'^logout/$', views.logout, name='logout'),
]
