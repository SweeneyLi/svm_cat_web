from django import template
import re

register = template.Library()


@register.filter()
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


@register.filter()
def field_type(field):
    return re.findall(r'"\w+"', str(field))[0].strip('"')


@register.filter()
def label_with_classes(value, arg):
    return value.label_tag(attrs={'class': arg})


@register.filter()
def widget_with_classes(value, arg):
    return value.as_widget(attrs={'class': arg})


@register.filter()
def remainder(value, oper):
    print(type(value))
    print(value)
    return value % int(oper)
