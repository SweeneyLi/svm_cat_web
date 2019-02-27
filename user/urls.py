from django.urls import re_path, path
from . import views

app_name = 'user'
urlpatterns = [
    path('register/', views.RegisterView.as_view(), name='register'),
    path('login/', views.LoginView.as_view(), name='login'),
    path('user/profile/', views.profile, name='profile'),
    path('user/profileUpdate/', views.ProfileUpdateView.as_view(), name='profile_update'),
    path('user/pwdchange/', views.PwdChangeView.as_view(), name='pwd_change'),
    path('logout/', views.logout, name='logout'),
]
