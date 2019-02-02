from django import forms
from picture.models import Picture


class HOGPicForm(forms.Form):
    pic_size = forms.CharField(initial='194,259')
    orientations = forms.IntegerField(initial='9')
    pixels_per_cell = forms.CharField(initial='8,8')
    cells_per_block = forms.CharField(initial='3,3')
    is_color = forms.BooleanField(initial=True, required=False)


class ChoosePicCategoryForm(forms.Form):
    test_pic = forms.ImageField()
    # category = forms.ModelChoiceField(queryset=Picture.objects.get(user_id=request))

    # test_category = forms.TypedChoiceField(
    #     choices=(
    #         (1, 'default')
    #     ),
    #     widget=forms.CheckboxSelectMultiple(),
    # )
