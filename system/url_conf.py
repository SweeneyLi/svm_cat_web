from django.urls import reverse, reverse_lazy

url_dict = {
    'profileUpdate': {
        'title': 'Profile Update',
        'next_name': 'My Profile',
        'next_url': reverse_lazy('user:profile'),

        'pic_url': '/static/img/LXH/1.jpg',
    },
    'pwdChange': {
        'title': 'Password Change',
        'next_name': 'My Profile',
        'next_url': reverse_lazy('user:profile'),

        'pic_url': '/static/img/LXH/1.jpg',
    },
    'register': {
        'title': 'Register',
        'next_name': 'Index',
        'next_url': reverse_lazy('system:index'),

        'pic_url': '/static/img/LXH/8.jpg',
    },
    'picUpload': {
        'title': 'Picture Upload',
        'next_name': 'Picture List',
        'next_url': reverse_lazy('picture:pic_list'),
        'pic_url': '/static/img/LXH/1.jpg',
    },
}
