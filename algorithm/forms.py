from django.conf import settings
from django import forms
from picture.models import Picture
from .models import SVMModel
from django.db.models import Count

import json

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
    test_pic = forms.ImageField(
        label='测试图片',  # label='Test Picture',
        widget=forms.FileInput(attrs={
            'class': 'custom-file-input'
        })
    )
    test_category_positive = forms.ModelChoiceField(Picture.objects.none())
    test_category_negative = forms.ModelChoiceField(Picture.objects.none())
    validation_size = forms.FloatField(label='数据集比例', initial=0.2)

    def __init__(self, user_id, *args, **kwargs):
        super(PrepareDataForm, self).__init__(*args, **kwargs)
        self.fields['test_category_positive'] = forms.ModelChoiceField(
            label='正确的图片类别',
            queryset=Picture.objects.filter(user_id=user_id).values('category').distinct().annotate(
                num_category=Count('category')
            ),
            empty_label="---------",
        )
        self.fields['test_category_negative'] = forms.ModelChoiceField(
            label='错误的图像类别',
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

        file = self.files.get('test_pic')
        ext = file.name.split('.')[-1].lower()
        if ext not in ["jpg", "jpeg", "png"]:
            self._errors = "上传文件必须为 jpg, jpeg, png格式！"
            # self._errors = "Only jpg, jpeg and png files are allowed."
            return False

        test_category_negative = self.data['test_category_negative']
        test_category_positive = self.data['test_category_positive']
        if test_category_positive == test_category_negative:
            self._errors = '正确的图片类别和错误的图片类别必须不同！'
            # self._errors = 'test_category_negative should diff from test_category_positive.'
            return False
        elif eval(test_category_positive)['num_category'] < 10:
            self._errors = '正确的图片类别必须最少包含10张图片！'
            # self._errors = 'test_category_positive should have al least ten pictures.'
            return False
        elif eval(test_category_negative)['num_category'] < 10:
            self._errors = '错误的图片类别必须最少包含10张图片！'
            # self._errors = 'test_category_negative should have al least ten pictures.'
            return False

        return True

    # def __str__(self):
    #     return u'{0}'.format(self.test_pic)


class HOGPicForm(forms.Form):
    pic_size = forms.CharField(
        label='图片大小',  # label='Picture Size',
        initial='194,259'
    )
    orientations = forms.IntegerField(
        label='bins个数',
        initial=9)
    pixels_per_cell = forms.CharField(
        label='每个cell的像素数',
        initial='16,16')
    cells_per_block = forms.CharField(
        label='每个BLOCK内cell分布',
        initial='2,2'
    )
    is_color = forms.BooleanField(
        label='是否选择颜色',
        initial=True,
        required=False
    )

    def is_valid(self):
        for field in ['pic_size', 'pixels_per_cell', 'cells_per_block']:
            # if not re.match(r'^\(\d+,\d+\)$', self.data[field]):
            if not is_tuple_positive(self.data[field]):
                self._errors = field + ' 格式错误 ! '
                # self._errors = 'The ' + field + ' is a wrong format ! '
                return False
        if int(self.data['orientations']) >= 0:
            return True
        else:
            self._errors = 'bin 的个数大于0且为整数！'
            # self._errors = 'The orientations should be int and bigger than 0'
            return False


class EvaluateAlgoritmForm(forms.Form):
    is_standard = forms.BooleanField(
        label='是否数据标准化',
        initial=False,
        required=False
    )

    algorithms = (('LR', 'LogisticRegression'), ('KNN', 'KNeighborsClassifier'),
                  ('CART', 'DecisionTreeClassifier'), ('NB', 'GaussianNB'))

    algorithm_list = forms.MultipleChoiceField(label='算法列表',
                                               choices=algorithms, widget=forms.CheckboxSelectMultiple(),
                                               required=False)


class SVMParameterForm(forms.Form):
    C = forms.CharField(
        label='惩罚系数',
        initial='0.1, 0.3, 0.5, 0.7, 0.9, 1.0',
        widget=forms.Textarea(attrs={'rows': 1, 'cols': 25})
    )
    kernel_list = (('linear', 'linear'), ('poly', 'poly'), ('rbf', 'rbf'), ('sigmoid', 'sigmoid'))
    kernel = forms.MultipleChoiceField(label='核函数',
                                       choices=kernel_list,
                                       widget=forms.CheckboxSelectMultiple(),
                                       required=True,
                                       )

    def is_valid(self):

        for a_C in self.data['C'].strip().split(','):
            if not is_float(a_C):
                self._errors = '惩罚系数 C 的格式有误! '
                # self._errors = 'The format of C is wrong! '
                return False
        if 'kernel' not in self.data:
            return False
        else:
            return True


class EnsembleParamsForm(forms.Form):
    C = forms.FloatField(label='惩罚系数')
    kernel_list = (('linear', 'linear'), ('poly', 'poly'), ('rbf', 'rbf'), ('sigmoid', 'sigmoid'))
    kernel = forms.ChoiceField(label='核函数',
                               choices=kernel_list
                               )

    ensemble_learning_list = (
        (0, "Don't use it"), ('AdaBoostClassifier', 'AdaBoostClassifier'), ('BaggingClassifier', 'BaggingClassifier'))

    ensemble_learning = forms.ChoiceField(
        label='集成算法',  # label='Ensemble learning',
        choices=ensemble_learning_list
    )
    n_estimators = forms.CharField(
        label='决策树个数',  # label='N estimators',
        initial='1,10')

    def __init__(self, user_id, *args, **kwargs):
        super(EnsembleParamsForm, self).__init__(*args, **kwargs)

        with open(settings.ALGORITHM_JSON_PATH, "r") as load_f:
            algorithm_info = json.load(load_f)

        if algorithm_info[str(user_id)]['model_para'].get('best_params', None):
            self.initial = {
                'C': algorithm_info[str(user_id)]['model_para']['best_params']['C'],
                'kernel': algorithm_info[str(user_id)]['model_para']['best_params']['kernel']
            }
        else:
            self.initial = {'C': 2.0, 'kernel': 'rbf'}

        self.fields['kernel'].widget.attrs.update(
            select_attrs
        )
        self.fields['ensemble_learning'].widget.attrs.update(
            select_attrs
        )

    def is_valid(self):

        for a_C in self.data['C'].strip().split(','):
            if not is_float(a_C):
                self._errors = '惩罚系数C的格式有误！'
                # self._errors = 'The format of C is wrong! '
                return False
        for a_est in self.data['n_estimators'].strip().split(','):
            if not a_est.isdigit():
                self._errors = '决策树数量输入格式有误！ '
                # self._errors = 'The format of n_estimatot is wrong! '
                return False
        if 'kernel' not in self.data:
            return False
        else:
            return True


class TrainLogForm(forms.Form):
    model_name = forms.ModelChoiceField(SVMModel.objects.none())
    train_category_positive = forms.ModelChoiceField(SVMModel.objects.none())
    train_category_negative = forms.ModelChoiceField(SVMModel.objects.none())
    validation_size = forms.FloatField(
        label='数据集比例',
        initial=0.2)

    def __init__(self, user_id, *args, **kwargs):
        super(TrainLogForm, self).__init__(*args, **kwargs)

        self.fields['model_name'] = forms.ModelChoiceField(
            label='模型名称',
            queryset=SVMModel.objects.filter(user_id=user_id).values('model_name').distinct().order_by('-update_time'),
            empty_label="---------", to_field_name="model_name"
        )

        self.fields['train_category_positive'] = forms.ModelChoiceField(
            label='训练正向数据集',
            queryset=Picture.objects.filter(user_id=user_id).values('category').distinct().annotate(
                num_category=Count('category')
            ).filter(user_id=user_id),
            empty_label="---------",
        )
        self.fields['train_category_negative'] = forms.ModelChoiceField(
            label='训练负向数据集',
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
            self._errors = '数据集比例应该位于0和1之间。'
            # self._errors = 'validation_size should between 0 and 1'
            return False
        if self.data['train_category_negative'] == self.data['train_category_positive']:
            self._errors = '正确的图片类别和错误的图片类别必须不同！'
            # self._errors = 'test_category_negative should diff from test_category_positive'
            return False
        else:
            return True


class CatIdentificationForm(forms.Form):
    file = forms.FileField(
        label='文件',
        widget=forms.ClearableFileInput(attrs={'multiple': True}))

    model_name = forms.ModelChoiceField(
        SVMModel.objects.none()
    )

    show_probility = forms.BooleanField(
        label='是否显示概率',
        initial=False, required=False)

    def __init__(self, user_id, *args, **kwargs):
        super(CatIdentificationForm, self).__init__(*args, **kwargs)

        # TODO: format the choice at page
        self.fields['model_name'] = forms.ModelChoiceField(
            label='模型名称',
            queryset=SVMModel.objects.filter(user_id=user_id).values('model_name').distinct().order_by('-update_time'),
            empty_label="---------",
        )

        self.fields['model_name'].widget.attrs.update(
            select_attrs
        )

    def is_valid(self):
        for file in self.files.getlist('file'):
            ext = file.name.split('.')[-1].lower()
            if ext not in ["jpg", "jpeg", "png"]:
                self._errors = "上传文件必须为 jpg, jpeg, png格式！"
                # self._errors = "Only jpg, jpeg and png files are allowed."
                return False
        return True
