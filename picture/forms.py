from django import forms


class FileUploadModelForm(forms.Form):
    category = forms.CharField()
    file = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))

    def is_valid(self):
        for file in self.files.getlist('file'):
            ext = file.name.split('.')[-1].lower()
            if ext not in ["jpg", "ipeg", "png"]:
                self._errors = "Only jpg, ipeg and png files are allowed."
                return False
        return True
