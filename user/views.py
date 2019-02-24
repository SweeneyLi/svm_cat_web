from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.models import User
from django.contrib import auth
from django.http import HttpResponseRedirect
from django.urls import reverse, reverse_lazy
from django.views.generic import FormView

from .forms import RegistrationForm, LoginForm, ProfileForm, PwdChangeForm
from .models import UserProfile


def register(request):
    if request.method == 'POST':

        form = RegistrationForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            email = form.cleaned_data['email']
            password = form.cleaned_data['password2']

            # 使用内置User自带create_user方法创建用户，不需要使用save()
            user = User.objects.create_user(username=username, password=password, email=email)

            # 如果直接使用objects.create()方法后不需要使用save()
            user_profile = UserProfile(user=user)
            user_profile.save()

            return HttpResponseRedirect("/accounts/login/")

    else:
        form = RegistrationForm()

    return render(request, 'user/registration.html', {'form': form})


def login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']

            filter_result = User.objects.filter(username__exact=username)
            if not filter_result:
                return render(request, 'user/login.html',
                              {'form': form, 'message': "This username does not exist. Please register first."})

            user = auth.authenticate(username=username, password=password)

            if not user:
                return render(request, 'user/login.html', {'form': form,
                                                           'message': 'Wrong password. Please try again.'})

            if user.is_superuser:
                auth.login(request, user)
                return HttpResponseRedirect("/admin")

            if user.is_active:
                auth.login(request, user)
                return HttpResponseRedirect(reverse('system:index'))

            else:
                # 登陆失败
                return render(request, 'user/login.html', {'form': form,
                                                           'message': 'Please try again.'})
    else:
        form = LoginForm()

    return render(request, 'user/login.html', {'form': form})


def profile(request):
    user_id = request.user.id
    userProfile = get_object_or_404(User, pk=user_id)
    return render(request, 'user/profile.html', {'user': userProfile})


class ProfileUpdateView(FormView):
    form_class = ProfileForm

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(request, 'system/common_form.html',
                      {'form': form,
                       'title': 'Profile Update',
                       'pic_url': '/static/img/LXH-1.jpg',
                       'next_url': reverse_lazy('user:profile'),
                       'next_name': 'Profile'
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
                       'title': 'Profile Update',
                       'pic_url': '/img/LXH-2.jpg',
                       'next_url': reverse_lazy('user:profile'),
                       'message': form.errors
                       })

    def form_valid(self, form, **kwargs):
        return redirect('user:profile')


def logout(request):
    auth.logout(request)
    return HttpResponseRedirect("/accounts/login/")


def pwd_change(request):
    user_id = request.user.id
    user = get_object_or_404(User, pk=user_id)
    if request.method == "POST":
        form = PwdChangeForm(request.POST)

        if form.is_valid():

            password = form.cleaned_data['old_password']
            username = user.username

            user = auth.authenticate(username=username, password=password)

            if user is not None and user.is_active:
                new_password = form.cleaned_data['password2']
                user.set_password(new_password)
                user.save()

                return redirect('user:logout')
                # form = LoginForm()
                # return render(request, 'user/login.html', {'form': form,
                #                                            'message': 'Please login again with new password!'})

            else:
                return render(request, 'user/pwd_change.html', {'form': form,
                                                                'user': user,
                                                                'message': 'Old password is wrong. Try again'})
    else:
        form = PwdChangeForm()
        return render(request, 'user/pwd_change.html', {'form': form, 'user': user})
