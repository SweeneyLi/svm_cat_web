from django import forms
from django.contrib.auth.models import User
import re


def email_check(email):
    pattern = re.compile(r"\"?([-a-zA-Z0-9.`?{}]+@\w+\.\w+)\"?")
    return re.match(pattern, email)


class RegistrationForm(forms.Form):
    username = forms.CharField(
        label='用户名',  # label='Username',
        max_length=50
    )
    email = forms.EmailField(
        label='邮箱',  # label='email',
    )
    password1 = forms.CharField(
        label='密码',  # label='Password',
        widget=forms.PasswordInput
    )
    password2 = forms.CharField(
        label='确认密码',  # label='Password Confirmation',
        widget=forms.PasswordInput
    )

    # Use clean methods to define custom validation rules

    def clean_username(self):
        username = self.cleaned_data.get('username')

        if len(username) < 6:
            raise forms.ValidationError("您的用户名至少六位！")
            # raise forms.ValidationError("Your username must be at least 6 characters long.")
        elif len(username) > 50:
            raise forms.ValidationError("您的用户名长度应小于50！")
            # raise forms.ValidationError("Your username is too long.")
        else:
            filter_result = User.objects.filter(username__exact=username)
            if len(filter_result) > 0:
                raise forms.ValidationError("该用户名已经存在了。")
                # raise forms.ValidationError("Your username already exists.")

        return username

    def clean_email(self):
        email = self.cleaned_data.get('email')

        if email_check(email):
            filter_result = User.objects.filter(email__exact=email)
            if len(filter_result) > 0:
                raise forms.ValidationError("您的邮件已经存在了！")
                # raise forms.ValidationError("Your email already exists.")
        else:
            raise forms.ValidationError("请输入一个合法的邮箱地址！")
            # raise forms.ValidationError("Please enter a valid email.")

        return email

    def clean_password1(self):
        password1 = self.cleaned_data.get('password1')

        if len(password1) < 6:
            raise forms.ValidationError("您的密码长度应大于6位。")
            # raise forms.ValidationError("Your password is too short.")
        elif len(password1) > 20:
            raise forms.ValidationError("您的密码长度应小于20位。")
            # raise forms.ValidationError("Your password is too long.")

        return password1

    def clean_password2(self):
        password1 = self.cleaned_data.get('password1')
        password2 = self.cleaned_data.get('password2')

        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("两次输入密码不一致，请重新输入！")
            # raise forms.ValidationError("Password mismatch. Please enter again.")

        return password2


class LoginForm(forms.Form):
    username = forms.CharField(
        label='用户名',  # label='Username',
        max_length=50, widget=forms.TextInput(attrs={'class': 'form-control'}))
    password = forms.CharField(
        label='密码',  # label='Password',
        widget=forms.PasswordInput(attrs={'class': 'form-control'}))

    # Use clean methods to define custom validation rules

    # def clean_username(self):
    #     username = self.cleaned_data.get('username')
    #
    #     if email_check(username):
    #         filter_result = User.objects.filter(email__exact=username)
    #         if not filter_result:
    #             raise forms.ValidationError("This email does not exist.")
    #     else:
    #         filter_result = User.objects.filter(username__exact=username)
    #         if not filter_result:
    #             raise forms.ValidationError("This username does not exist. Please register first.")
    #
    #     return username


class ProfileForm(forms.Form):
    email = forms.CharField(
        label='邮箱',  # label='email',
        max_length=50, required=False
    )
    first_name = forms.CharField(
        label='姓',  # label='First Name',
        max_length=50, required=False)
    last_name = forms.CharField(
        label='名字',  # label='Last Name',
        max_length=50, required=False)
    org = forms.CharField(
        label='组织',  # label='Organization',
        max_length=50, required=False)
    telephone = forms.CharField(
        label='电话',  # label='Telephone',
        max_length=50, required=False)


class PwdChangeForm(forms.Form):
    old_password = forms.CharField(
        label='旧密码',  # label='Old password',
        widget=forms.PasswordInput)

    password1 = forms.CharField(
        label='新密码',  # label='New Password',
        widget=forms.PasswordInput)
    password2 = forms.CharField(
        label='确认密码',  # label='Password Confirmation',
        widget=forms.PasswordInput)

    # Use clean methods to define custom validation rules

    def clean_password1(self):
        password1 = self.cleaned_data.get('password1')

        if len(password1) < 6:
            raise forms.ValidationError("密码长度最少六位！")
            # raise forms.ValidationError("Your password is too short.")
        elif len(password1) > 20:
            raise forms.ValidationError("密码长度最长20位！")
            # raise forms.ValidationError("Your password is too long.")

        return password1

    def clean_password2(self):
        password1 = self.cleaned_data.get('password1')

        password2 = self.cleaned_data.get('password2')

        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("两次输入密码不一致，请重新输入！")
            # raise forms.ValidationError("Password mismatch. Please enter again.")

        return password2
