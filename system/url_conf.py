from django.urls import reverse, reverse_lazy

url_dict = {
    'profileUpdate': {
        'title': 'Profile Update',
        'next_url': reverse_lazy('user:profile'),
        'next_name': 'My Profile',
        'pic_url': '/static/img/LXH-1.jpg',
    },
    'pwdChange': {
        'title': 'PassWord Change',
        'next_url': reverse_lazy('user:profile'),
        'next_name': 'My Profile',
        'pic_url': '/static/img/LXH-1.jpg',
    },
}
