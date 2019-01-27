from django.http import HttpResponseRedirect
from django.urls import reverse
from django.utils.deprecation import MiddlewareMixin


class UserAuthMiddle(MiddlewareMixin):

    def process_request(self, request):

        not_need_login = ['/accounts/login/', '/accounts/register/', '/index/']

        if request.path in not_need_login or request.user.is_authenticated:
            return None
        else:
            print('Not authenticated!!!')
            return HttpResponseRedirect(reverse('user:login'))
