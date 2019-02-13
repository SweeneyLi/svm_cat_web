from django.http import HttpResponseRedirect
from django.views.generic import DetailView, ListView
from django.views.generic.edit import FormView, DeleteView
from django.urls import reverse_lazy

from PIL import Image
from os import path
from datetime import datetime, timedelta

from .forms import FileUploadModelForm
from .models import Picture
from django.conf import settings

import shutil


class PicList(ListView):
    context_object_name = 'picture_list'

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
    template_name = 'picture/picture_upload.html'
    success_url = reverse_lazy('picture:pic_list')

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        files = request.FILES.getlist('file')
        user_id = self.request.user.id

        # TODO: safe
        category = self.request.POST['category']

        if form.is_valid():
            for f in files:
                im = Image.open(f)
                pic_size = im.size
                pic_name = f.name
                the_path = path.join(settings.MEDIA_ROOT, 'upload_images', str(user_id), category, pic_name)
                if path.exists(the_path):
                    upload_pic_name, ext = pic_name.split('.')[0], path.splitext(pic_name)[1]
                    time = (datetime.now() + timedelta(hours=8)).strftime("%Y-%m-%d_%H:%M:%S.")
                    pic_name = upload_pic_name + '_' + time + ext
                pic = Picture(user_id=user_id, pic_name=pic_name, path=f, category=category, pic_size=pic_size)
                pic.save()
            return self.form_valid(form)
        else:
            return self.form_invalid(form)


class PicDeleteView(DeleteView):
    model = Picture
    success_url = reverse_lazy('picture:pic_list')
    template_name = 'picture/picture_delete.html'

    def get(self, request, *args, **kwargs):
        cate_pic = self.get_object()
        context ={
            'category': cate_pic[0].category,
            'cate_num': len(cate_pic),
            'pic_list': [pic.pic_name for pic in cate_pic]
        }

        return self.render_to_response(context)

    def get_queryset(self):
        return self.model.objects.filter(
            user_id=self.request.user.id
        )

    def get_object(self, queryset=None):
        queryset = self.get_queryset()

        category = self.kwargs['pk']

        obj = queryset.filter(category=category)
        return obj

    def delete(self, request, *args, **kwargs):
        response = super(PicDeleteView, self).delete(request, *args, **kwargs)

        user_id = self.request.user.id
        category = self.kwargs['pk']

        category_path = path.join(settings.MEDIA_ROOT, 'upload_images', str(user_id), category)
        shutil.rmtree(category_path)

        return response
