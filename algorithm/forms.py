from django import forms


class PicProcessingForm(forms.Form):

    pic_size = forms.CharField(initial='194,259')
    test_pic = forms.ImageField()
    orientations = forms.IntegerField(initial='9')
    pixels_per_cell = forms.CharField(initial='8,8')
    cells_per_block = forms.CharField(initial='3,3')
    is_color = forms.BooleanField(initial='True')

    # def clean_file(self):
    #     file = self.cleaned_data['file']
    #     ext = file.name.split('.')[-1].lower()
    #     if ext not in ["jpg", "ipeg", "png"]:
    #         raise forms.ValidationError("Only jpg, ipeg and png files are allowed.")
    #     return file

    # def __init__(self, *args, **kwargs):
    #     super(PicProcessingForm, self).__init__(*args, **kwargs)
    #     self.fields["test_pic"].widget.choices = Picture.objects.filter(user_id=request.user.id).values_list("pic_name")

