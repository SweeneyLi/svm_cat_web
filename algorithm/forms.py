from django import forms
from picture.models import Picture
from .models import SVMModel
from django.db.models import Count
from django.conf import settings
import json
import re

select_attrs = {"class": "browser-default custom-select custom-select-lg mb-3"}


def is_float(str):
    try:
        a = float(str)
    except Exception:
        return False
    if a == 0:
        return False
    else:
        return True


def is_tuple_positive(str):
    try:
        temp = str.split(',')
        if len(temp) == 2:
            for i in temp:
                if int(i) <= 0:
                    return False
        else:
            return False
    except Exception:
        return False
    return True


class PrepareDataForm(forms.Form):
    test_pic = forms.ImageField(label='Test Picture', widget=forms.FileInput(attrs={
        'class': 'custom-file-input'
    }))
    test_category_positive = forms.ModelChoiceField(Picture.objects.none())
    test_category_negative = forms.ModelChoiceField(Picture.objects.none())
    validation_size = forms.FloatField(label='Validation Size', initial=0.2)

    def __init__(self, user_id, *args, **kwargs):
        super(PrepareDataForm, self).__init__(*args, **kwargs)
        self.fields['test_category_positive'] = forms.ModelChoiceField(
            queryset=Picture.objects.filter(user_id=user_id).values('category').distinct().annotate(
                num_category=Count('category')
            ).filter(user_id=user_id),
            empty_label="---------",
        )
        self.fields['test_category_negative'] = forms.ModelChoiceField(
            queryset=Picture.objects.filter(user_id=user_id).values('category').distinct().annotate(
                num_category=Count('category')
            ),
            empty_label="---------",
        )

        self.fields['test_category_positive'].widget.attrs.update(
            select_attrs
        )
        self.fields['test_category_negative'].widget.attrs.update(
            select_attrs
        )

    def is_valid(self):

        if self.data['test_category_negative'] == self.data['test_category_positive']:
            self._errors = 'test_category_negative should diff from test_category_positive'
            return False
        else:
            return True


class HOGPicForm(forms.Form):
    pic_size = forms.CharField(label='Picture Size', initial='194,259')
    orientations = forms.IntegerField(initial=9)
    pixels_per_cell = forms.CharField(initial='16,16')
    cells_per_block = forms.CharField(initial='2,2')
    is_color = forms.BooleanField(initial=True, required=False)

    def is_valid(self):
        # TODO:judege
        for field in ['pic_size', 'pixels_per_cell', 'cells_per_block']:
            # if not re.match(r'^\(\d+,\d+\)$', self.data[field]):
            if not is_tuple_positive(self.data[field]):
                self._errors = 'The ' + field + ' is a wrong format ! '
                return False
        if int(self.data['orientations']) >= 0:
            return True
        else:
            self._errors = 'The orientations should be int and bigger than 0'
            return False


class EvaluateAlgoritmForm(forms.Form):
    is_standard = forms.BooleanField(initial=False, required=False)

    algorithms = (('LR', 'LogisticRegression'), ('KNN', 'KNeighborsClassifier'),
                  ('CART', 'DecisionTreeClassifier'), ('NB', 'GaussianNB'))

    algorithm_list = forms.MultipleChoiceField(label='algorithm_list',
                                               choices=algorithms, widget=forms.CheckboxSelectMultiple())


class SVMParameterForm(forms.Form):
    C = forms.CharField(initial='0.1, 0.3, 0.5, 0.7, 0.9, 1.0', widget=forms.Textarea(attrs={'rows': 1, 'cols': 25}))
    kernel_list = (('linear', 'linear'), ('poly', 'poly'), ('rbf', 'rbf'), ('sigmoid', 'sigmoid'))
    kernel = forms.MultipleChoiceField(label='kernel',
                                       choices=kernel_list,
                                       widget=forms.CheckboxSelectMultiple(),
                                       required=True,
                                       )

    def is_valid(self):

        for a_C in self.data['C'].strip().split(','):
            if not is_float(a_C):
                self._errors = 'The format of C is wrong! '
                return False
        if 'kernel' not in self.data:
            return False
        else:
            return True


class EnsembleParamsForm(forms.Form):
    C = forms.FloatField(label='C')
    kernel_list = (('linear', 'linear'), ('poly', 'poly'), ('rbf', 'rbf'), ('sigmoid', 'sigmoid'))
    kernel = forms.ChoiceField(label='kernel',
                               choices=kernel_list
                               )

    ensemble_learning_list = ((0, "Don't use it"), ('AdaBoostClassifier', 'AdaBoostClassifier'), ('BaggingClassifier', 'BaggingClassifier'))

    ensemble_learning = forms.ChoiceField(label='Ensemble learning',
                                          choices=ensemble_learning_list)
    n_estimators = forms.CharField(label='N estimators', initial='1,10')

    def __init__(self, user_id, *args, **kwargs):
        super(EnsembleParamsForm, self).__init__(*args, **kwargs)

        with open(settings.ALGORITHM_JSON_PATH, "r") as load_f:
            algorithm_info = json.load(load_f)

        self.initial = {
            'C': algorithm_info[str(user_id)]['model_para']['best_params']['C'],
            'kernel': algorithm_info[str(user_id)]['model_para']['best_params']['kernel']
        }

        self.fields['kernel'].widget.attrs.update(
            select_attrs
        )
        self.fields['ensemble_learning'].widget.attrs.update(
            select_attrs
        )

    def is_valid(self):

        for a_C in self.data['C'].strip().split(','):
            if not is_float(a_C):
                self._errors = 'The format of C is wrong! '
                return False
        for a_est in self.data['n_estimators'].strip().split(','):
            if not a_est.isdigit():
                self._errors = 'The format of n_estimatot is wrong! '
                return False
        if 'kernel' not in self.data:
            return False
        else:
            return True


class TrainLogForm(forms.Form):
    model_name = forms.ModelChoiceField(SVMModel.objects.none())
    train_category_positive = forms.ModelChoiceField(SVMModel.objects.none())
    train_category_negative = forms.ModelChoiceField(SVMModel.objects.none())
    validation_size = forms.FloatField(initial=0.2)

    def __init__(self, user_id, *args, **kwargs):
        super(TrainLogForm, self).__init__(*args, **kwargs)

        self.fields['model_name'] = forms.ModelChoiceField(
            queryset=SVMModel.objects.filter(user_id=user_id).values('model_name').distinct().order_by('-update_time'),
            empty_label="---------", to_field_name="model_name"
        )

        self.fields['train_category_positive'] = forms.ModelChoiceField(
            queryset=Picture.objects.filter(user_id=user_id).values('category').distinct().annotate(
                num_category=Count('category')
            ).filter(user_id=user_id),
            empty_label="---------",
        )
        self.fields['train_category_negative'] = forms.ModelChoiceField(
            queryset=Picture.objects.filter(user_id=user_id).values('category').distinct().annotate(
                num_category=Count('category')
            ),
            empty_label="---------",
        )

        self.fields['model_name'].widget.attrs.update(
            select_attrs
        )
        self.fields['train_category_positive'].widget.attrs.update(
            select_attrs
        )
        self.fields['train_category_negative'].widget.attrs.update(
            select_attrs
        )

    def is_valid(self):
        if not (0.0 <= float(self.data['validation_size']) < 1.0):
            self._errors = 'validation_size should between 0 and 1'
            return False
        if self.data['train_category_negative'] == self.data['train_category_positive']:
            self._errors = 'test_category_negative should diff from test_category_positive'
            return False
        else:
            return True


class CatIdentificationForm(forms.Form):
    file = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))

    model_name = forms.ModelChoiceField(SVMModel.objects.none())

    show_probility = forms.BooleanField(initial=False, required=False)

    def __init__(self, user_id, *args, **kwargs):
        super(CatIdentificationForm, self).__init__(*args, **kwargs)

        # TODO: format the choice at page
        self.fields['model_name'] = forms.ModelChoiceField(
            queryset=SVMModel.objects.filter(user_id=user_id).values('model_name').distinct().order_by('-update_time'),
            empty_label="---------",
        )

        self.fields['model_name'].widget.attrs.update(
            select_attrs
        )
