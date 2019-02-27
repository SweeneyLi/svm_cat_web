from django.urls import reverse, reverse_lazy

url_dict = {
    'profileUpdate': {
        'title': 'Profile Update',
        'next_name': 'My Profile',
        'next_url': reverse_lazy('user:profile'),
        'pic_url': None,
    },
    'pwdChange': {
        'title': 'Password Change',
        'next_name': 'My Profile',
        'next_url': reverse_lazy('user:profile'),

        'pic_url': None,
    },
    'register': {
        'title': 'Register',
        'next_name': 'Index',
        'next_url': reverse_lazy('system:index'),

        'pic_url': None,
    },
    'picUpload': {
        'title': 'Picture Upload',
        'next_name': 'Picture List',
        'next_url': reverse_lazy('picture:pic_list'),
        'pic_url': None,
    },
    'trainSVMModel': {
        'title': 'Train SVM Model',
        'next_name': 'Model List',
        'next_url': reverse_lazy('alogrithm:model_list'),
        'next_name_2': 'Cat Identification',
        'next_url_2': reverse_lazy('alogrithm:cat_identification'),
        'pic_url': None,
    },
    'catIdentification': {
        'title': 'Cat Identification',
        'next_name': 'Model List',
        'next_url': reverse_lazy('alogrithm:model_list'),
        'pic_url': None,
    },
}
