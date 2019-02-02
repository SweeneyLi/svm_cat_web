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
    test_category_positive = forms.ModelChoiceField(Picture.objects.none())
    test_category_negative = forms.ModelChoiceField(Picture.objects.none())

    def __init__(self, user_id, *args, **kwargs):
        super(ChoosePicCategoryForm, self).__init__(*args, **kwargs)
        self.fields['test_category_positive'] = forms.ModelChoiceField(
            queryset=Picture.objects.filter(user_id=user_id).values('category').distinct(),
            empty_label="请至少选择一个",
        )
        self.fields['test_category_negative'] = forms.ModelChoiceField(
            queryset=Picture.objects.filter(user_id=user_id).values('category').distinct(),
            empty_label="请至少选择一个",
        )

    def is_valid(self):
        # run the parent validation first
        valid = super(ChoosePicCategoryForm, self).is_valid()

        if not valid:
            return valid
        elif self.fields['test_category_negative'] == self.fields['test_category_positive']:
            self._errors['same_category'] = 'test_category_negative should diff from test_category_positive'
            return False
        else:
            return valid
