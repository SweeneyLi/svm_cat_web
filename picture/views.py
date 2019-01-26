from django.views.generic import DetailView, ListView
from django.views.generic.edit import FormView
from django.urls import reverse_lazy

from PIL import Image
from os import path
import datetime

from .forms import FileUploadModelForm
from .models import Picture
from django.conf import settings


class PicList(ListView):
    context_object_name = 'latest_picture_list'

    template_name = 'picture/picture_list.html'

    def get_queryset(self):
        return Picture.objects.all(). \
            filter(user_id=self.request.user.id).order_by('category', '-upload_date')


class PicDetail(DetailView):
    model = Picture
    Context_object_name = 'picture_detail'

    template_name = 'picture/picture_detail.html'


class PicUpload(FormView):
    form_class = FileUploadModelForm
    template_name = 'picture/picture_form.html'
    success_url = reverse_lazy('picture:pic_list')

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        files = request.FILES.getlist('file')
        user_id = self.request.user.id

        category = self.request.POST['category']
        category = category.replace('/', '')

        if form.is_valid():
            for f in files:
                im = Image.open(f)
                pic_size = im.size
                pic_name = f.name
                the_path = path.join(settings.MEDIA_ROOT, str(user_id), category, pic_name)
                if path.exists(the_path):
                    upload_pic_name, ext = pic_name.split('.')[0], pic_name.split('.')[-1]
                    pic_name = upload_pic_name + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.") + ext
                pic = Picture(user_id=user_id, pic_name=pic_name, path=f, category=category, pic_size=pic_size)
                pic.save()
            return self.form_valid(form)
        else:
            return self.form_invalid(form)
