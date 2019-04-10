from django.urls import reverse, reverse_lazy

url_dict = {
    'login': {
        'title': '登录',  # 'Please Login',
        'next_name': '注册',  # 'Register',
        'next_url': reverse_lazy('user:register'),
        'pic_url': None,
    },
    'profileUpdate': {
        'title': '信息更改',  # 'Profile Update',
        'next_name': '',  # 'My Profile',
        'next_url': reverse_lazy('user:profile'),
        'pic_url': None,
    },
    'pwdChange': {
        'title': '更改密码',  # 'Password Change',
        'next_name': '我的信息',  # 'My Profile',
        'next_url': reverse_lazy('user:profile'),

        'pic_url': None,
    },
    'register': {
        'title': '注册',  # 'Register',
        'next_name': '登录',  # 'Login',
        'next_url': reverse_lazy('user:login'),
        'pic_url': None,
    },
    'picUpload': {
        'title': '图片上传',  # 'Picture Upload',
        'next_name': '图片库',   # 'Picture List',
        'next_url': reverse_lazy('picture:pic_list'),
        'next_name_2': '建立模型', # 'Start Make Model',
        'next_url_2': reverse_lazy('alogrithm:prepare_data'),
        'pic_url': None,
    },
    'trainSVMModel': {
        'title': '训练模型',  # 'Train SVM Model',
        'next_name': '模型库',  # 'Model List',
        'next_url': reverse_lazy('alogrithm:model_list'),
        'next_name_2': '猫咪识别', # 'Cat Identification',
        'next_url_2': reverse_lazy('alogrithm:cat_identification'),
        'pic_url': None,
    },
    'catIdentification': {
        'title': '猫咪识别',  # 'Cat Identification',
        'next_name': '模型库',   # 'Model List',
        'next_url': reverse_lazy('alogrithm:model_list'),
        'pic_url': None,
    },
}
