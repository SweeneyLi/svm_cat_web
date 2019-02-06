from django import forms
from picture.models import Picture
from django.db.models import Count

# TODO: all the valid of forms

class ChoosePicCategoryForm(forms.Form):
    test_pic = forms.ImageField()
    test_category_positive = forms.ModelChoiceField(Picture.objects.none())
    test_category_negative = forms.ModelChoiceField(Picture.objects.none())
    validation_size = forms.FloatField(initial=0.2)

    def __init__(self, user_id, *args, **kwargs):
        super(ChoosePicCategoryForm, self).__init__(*args, **kwargs)
        self.fields['test_category_positive'] = forms.ModelChoiceField(
            queryset=Picture.objects.filter(user_id=user_id).values('category').distinct().annotate(
                num_category=Count('category')
            ).filter(user_id=user_id),
            empty_label="请至少选择一个",
        )
        self.fields['test_category_negative'] = forms.ModelChoiceField(
            queryset=Picture.objects.filter(user_id=user_id).values('category').distinct().annotate(
                num_category=Count('category')
            ),
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


class HOGPicForm(forms.Form):
    pic_size = forms.CharField(initial='(194,259)')
    orientations = forms.IntegerField(initial='9')
    pixels_per_cell = forms.CharField(initial='(16,16)')
    cells_per_block = forms.CharField(initial='(2,2)')
    is_color = forms.BooleanField(initial=True, required=False)


class ContrastAlgorithmForm(forms.Form):
    is_standard = forms.BooleanField(initial=False, required=False)

    # algorithms = ('LogisticRegression', 'KNeighborsClassifier',
    #               'DecisionTreeClassifier', 'GaussianNB')
    algorithms = (('1', 'svm'), ('2', 'asdf'))

    contrast_algorithm = forms.MultipleChoiceField(label='contrast_algorithmthms',
                                                   choices=algorithms, widget=forms.CheckboxSelectMultiple())


class SVMParameterForm(forms.Form):
    c = forms.CharField(initial='0.1, 0.3, 0.5, 0.7, 0.9, 1.0')
    kernel_list = (('linear', 'linear'), ('poly', 'poly'), ('rbf', 'rbf'), ('sigmoid', 'sigmoid'))
    kernel = forms.MultipleChoiceField(label='kernel',
                                       choices=kernel_list,
                                       widget=forms.CheckboxSelectMultiple()
                                       )
