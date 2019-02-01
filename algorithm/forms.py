from django import forms
from picture.models import Picture

class PicProcessingForm(forms.Form):
    pic_size = forms.CharField(initial='194,259')
    orientations = forms.IntegerField(initial='9')
    pixels_per_cell = forms.CharField(initial='8,8')
    cells_per_block = forms.CharField(initial='3,3')
    is_color = forms.BooleanField(initial='True')


class ChoosePicCategoryForm(forms.Form):
    test_pic = forms.ImageField()
    # category = forms.ModelChoiceField(queryset=Picture)
