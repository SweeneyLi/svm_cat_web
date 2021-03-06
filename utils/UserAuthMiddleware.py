from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse
from django.utils.deprecation import MiddlewareMixin


class UserAuthMiddle(MiddlewareMixin):

    def process_request(self, request):

        not_need_login = ['/accounts/login/', '/accounts/register/', '/index/', '/admin/', '/favicon.ico', '/test/', ]

        if request.path in not_need_login or request.user.is_authenticated:
            return None
        else:
            print(str(request.path) + '-' * 10 + 'Not authenticated!!!')
            return HttpResponseRedirect(reverse('user:login'))
