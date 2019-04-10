from django import forms


class FileUploadModelForm(forms.Form):
    category = forms.CharField(
        label='类别', required=True
    )
    file = forms.FileField(
        label='图片',
        widget=forms.ClearableFileInput(attrs={'multiple': True}))

    def is_valid(self):
        if self.data['category'] == '':
            return False
        for i in self.data['category']:
            if not i.isalnum():
                self._errors = "图片类别命名只能包含英文和数字！"
                # self._errors = "The name of category only conatin the number and letter ！"
                return False

        for file in self.files.getlist('file'):
            ext = file.name.split('.')[-1].lower()
            if ext not in ["jpg", "jpeg", "png"]:
                # self._errors = "Only jpg, jpeg and png files are allowed."
                self._errors = "上传文件格式为jpg, jpeg 或者 png！"
                return False
        return True
