from django import template

register = template.Library()


def num_to_English(num):
    num_English_dict = {
        1: 'First',
        2: 'Second',
        3: 'Third',
        4: 'Forth',
        5: 'Fitrh',
        6: 'Sixth',
        7: 'Seventh'
    }
    return num_English_dict[int(num)]


register.filter(num_to_English)

#
# @register.filter(name='get_pic_name')
# def get_pic_name(path):
#     return path.split('/')[-1]
