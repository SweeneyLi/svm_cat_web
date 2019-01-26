from django import forms


class FileUploadModelForm(forms.Form):
    category = forms.CharField()
    file = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))

    def clean_file(self):
        file = self.cleaned_data['file']
        ext = file.name.split('.')[-1].lower()
        if ext not in ["jpg", "ipeg", "png"]:
            raise forms.ValidationError("Only jpg, ipeg and png files are allowed.")
        return file
