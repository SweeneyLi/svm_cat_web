from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.models import User
from django.contrib import auth
from django.http import HttpResponseRedirect
from django.urls import reverse, reverse_lazy
from django.views.generic import FormView
from django.conf import settings

from .forms import RegistrationForm, LoginForm, ProfileForm, PwdChangeForm
from .models import UserProfile
from system.url_conf import *

import json


class RegisterView(FormView):
    form_class = RegistrationForm
    view_name = 'register'

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(request, 'system/common_form.html',
                      {'form': form,
                       'url_info': url_dict[self.view_name],
                       })

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form, **kwargs)
        else:
            return self.form_invalid(form, **kwargs)

    def form_invalid(self, form, **kwargs):
        return render(self.request, 'system/common_form.html',
                      {'form': form,
                       'url_info': url_dict[self.view_name],
                       })

    def form_valid(self, form, **kwargs):

        username = form.cleaned_data['username']
        email = form.cleaned_data['email']
        password = form.cleaned_data['password2']

        user = User.objects.create_user(username=username, password=password, email=email)

        user_profile = UserProfile(user=user)
        user_profile.save()

        # algorithm_info_json initial
        with open(settings.ALGORITHM_JSON_PATH, "r") as load_f:
            algorithm_info = json.load(load_f)
        user_id = user.id
        algorithm_info[user_id] = {}

        algorithm_info_keys = algorithm_info[user_id].keys()
        for key in ['pic_para', 'data_para', 'model_para', 'ensemble_para']:
            if key not in algorithm_info_keys:
                algorithm_info[user_id][key] = {}

        with open(settings.ALGORITHM_JSON_PATH, 'w') as f:
            json.dump(algorithm_info, f)

        return HttpResponseRedirect("/accounts/login/")


class LoginView(FormView):
    form_class = LoginForm
    view_name = 'login'

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(request, 'system/common_form.html',
                      {'form': form,
                       'url_info': url_dict[self.view_name],
                       })

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form, **kwargs)
        else:
            return self.form_invalid(form, **kwargs)

    def form_invalid(self, form, **kwargs):
        return render(self.request, 'system/common_form.html',
                      {'form': form,
                       'url_info': url_dict[self.view_name],
                       })

    def form_valid(self, form, **kwargs):
        username = form.cleaned_data['username']
        password = form.cleaned_data['password']

        filter_result = User.objects.filter(username__exact=username)
        if not filter_result:
            return render(self.request, 'system/common_form.html',
                          {'form': form,
                           'url_info': url_dict[self.view_name],
                           'message': "This username does not exist. Please register first."
                           })

        user = auth.authenticate(username=username, password=password)

        if not user:
            return render(self.request, 'system/common_form.html',
                          {'form': form,
                           'url_info': url_dict[self.view_name],
                           'message': 'Wrong password. Please try again.'
                           })

        if user.is_superuser:
            auth.login(self.request, user)
            return HttpResponseRedirect("/admin")

        if user.is_active:
            auth.login(self.request, user)

            return HttpResponseRedirect(reverse('system:index'))

        else:
            return render(self.request, 'system/common_form.html',
                          {'form': form,
                           'url_info': url_dict[self.view_name],
                           'message': 'Please try again.'
                           })


def profile(request):
    user_id = request.user.id
    userProfile = get_object_or_404(User, pk=user_id)
    return render(request, 'user/profile.html', {'user': userProfile})


class ProfileUpdateView(FormView):
    form_class = ProfileForm
    view_name = 'profileUpdate'

    def get_form_kwargs(self):
        kwargs = super(ProfileUpdateView, self).get_form_kwargs()
        user_id = str(self.request.user.id)
        user = User.objects.get(id=user_id)

        kwargs['initial']['email'] = user.email
        kwargs['initial']['first_name'] = user.first_name
        kwargs['initial']['last_name'] = user.last_name
        kwargs['initial']['org'] = user.profile.org
        kwargs['initial']['telephone'] = user.profile.telephone
        return kwargs

    def get(self, request, *args, **kwargs):
        form = self.get_form()

        return render(request, 'system/common_form.html',
                      {'form': form,
                       'url_info': url_dict[self.view_name]
                       })

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form, **kwargs)
        else:
            return self.form_invalid(form, **kwargs)

    def form_invalid(self, form, **kwargs):
        return render(self.request, 'system/common_form.html',
                      {'form': form,
                       'url_info': url_dict[self.view_name],
                       })

    def form_valid(self, form, **kwargs):
        user_id = str(self.request.user.id)
        user = User.objects.get(id=user_id)
        user.email = form.cleaned_data['email']
        user.first_name = form.cleaned_data['first_name']
        user.last_name = form.cleaned_data['last_name']
        user.save()

        user.profile.org = form.cleaned_data['org']
        user.profile.telephone = form.cleaned_data['telephone']
        user.profile.save()
        return redirect('user:profile')


class PwdChangeView(FormView):
    form_class = PwdChangeForm
    view_name = 'pwdChange'

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(request, 'system/common_form.html',
                      {'form': form,
                       'url_info': url_dict[self.view_name],
                       })

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form, **kwargs)
        else:
            return self.form_invalid(form, **kwargs)

    def form_invalid(self, form, **kwargs):
        return render(self.request, 'system/common_form.html',
                      {'form': form,
                       'url_info': url_dict[self.view_name],
                       })

    def form_valid(self, form, **kwargs):
        user_id = self.request.user.id
        user = get_object_or_404(User, pk=user_id)

        password = form.cleaned_data['old_password']
        username = user.username

        user = auth.authenticate(username=username, password=password)

        if user is not None and user.is_active:
            new_password = form.cleaned_data['password2']
            user.set_password(new_password)
            user.save()

            return redirect('user:logout')
        else:
            form._errors = 'Origin password is wrong. Try again'
            return render(self.request, 'system/common_form.html',
                          {'form': form,
                           'url_info': url_dict[self.view_name],
                           # 'message': 'Old password is wrong. Try again'
                           })


def logout(request):
    auth.logout(request)
    return HttpResponseRedirect("/accounts/login/")
