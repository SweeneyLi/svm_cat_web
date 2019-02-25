from django.urls import reverse, reverse_lazy

url_dict = {
    'profileUpdate': {
        'title': 'Profile Update',
        'next_url': reverse_lazy('user:profile'),
        'next_name': 'My Profile',
        'pic_url': '/static/img/LXH/1.jpg',
    },
    'pwdChange': {
        'title': 'Password Change',
        'next_url': reverse_lazy('user:profile'),
        'next_name': 'My Profile',
        'pic_url': '/static/img/LXH/1.jpg',
    },
    'register': {
        'title': 'Register',
        'next_url': reverse_lazy('system:index'),
        'next_name': 'Index',
        'pic_url': '/static/img/LXH/8.jpg',
    },
}
