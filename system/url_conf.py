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
    'trainSVMModel': {
        'title': 'Train SVM Model',
        'next_name': 'Model List',
        'next_url': reverse_lazy('alogrithm:model_list'),
        'next_name_2': 'Model List',
        'next_url_2': reverse_lazy('alogrithm:cat_identification'),
        'pic_url': '/static/img/LXH/1.jpg',
    },
    'catIdentification': {
        'title': 'Cat Identification',
        'next_name': 'Model List',
        'next_url': reverse_lazy('alogrithm:model_list'),
        'pic_url': '/static/img/LXH/1.jpg',
    },
}
