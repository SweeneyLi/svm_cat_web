from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.views.generic import DetailView, ListView
from django.views.generic.edit import FormView, DeleteView
from django.urls import reverse_lazy

from .forms import FileUploadModelForm
from .models import Picture
from django.conf import settings
from system.url_conf import url_dict

from datetime import datetime, timedelta
from PIL import Image
import shutil
import os


class PicListView(ListView):
    context_object_name = 'picture_list'

    template_name = 'picture/picture_list.html'

    def get_queryset(self):
        return Picture.objects.all(). \
            filter(user_id=self.request.user.id).order_by('category', '-upload_date')


# class PicDetailView(DetailView):
#     model = Picture
#     Context_object_name = 'picture_detail'
    # template_name = 'picture/picture_detail.html'


class PicUploadView(FormView):
    form_class = FileUploadModelForm
    view_name = 'picUpload'

    def get(self, request, *args, **kwargs):
        form = self.get_form()
        return render(request, 'system/common_form.html',
                      {'form': form,
                       'url_info': url_dict[self.view_name],
                       })

    def post(self, request, *args, **kwargs):
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form, **kwargs)
        else:
            return self.form_invalid(form, **kwargs)

    def form_invalid(self, form, **kwargs):
        return render(self.request, 'system/common_form.html',
                      {'form': form,
                       'url_info': url_dict[self.view_name],
                       })

    def form_valid(self, form, **kwargs):
        files = self.request.FILES.getlist('file')
        user_id = self.request.user.id

        # TODO: safe
        category = self.request.POST['category']

        for f in files:
            im = Image.open(f)
            pic_size = im.size
            pic_name = f.name
            the_path = os.path.join(settings.MEDIA_ROOT, 'upload_images', str(user_id), category, pic_name)
            if os.path.exists(the_path):
                upload_pic_name, ext = pic_name.split('.')[0], os.path.splitext(pic_name)[1]
                time = (datetime.now() + timedelta(hours=8)).strftime("%Y-%m-%d_%H:%M:%S.")
                pic_name = upload_pic_name + '_' + time + ext
            pic = Picture(user_id=user_id, pic_name=pic_name, path=f, category=category, pic_size=pic_size)
            pic.save()

        s = '' if len(files) == 1 else 's'
        message = "You have uploaded " + str(len(files)) + " image" + s + " to the " + category + " successfully."
        return render(self.request, 'system/common_form.html',
                      {'form': form,
                       'url_info': url_dict[self.view_name],
                       'message': message
                       })


class PicDeleteView(DeleteView):
    model = Picture
    success_url = reverse_lazy('picture:pic_list')

    def get_queryset(self):
        return self.model.objects.filter(
            user_id=self.request.user.id
        )

    def get_object(self, queryset=None):
        queryset = self.get_queryset()

        parameter = self.kwargs['pk']
        if parameter.isdigit():
            obj = queryset.filter(id=parameter)
        else:
            obj = queryset.filter(category=parameter)
        return obj

    def get(self, request, *args, **kwargs):
        return self.delete(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):

        user_id = self.request.user.id
        parameter = self.kwargs['pk']

        if parameter.isdigit():
            queryset = self.get_queryset()
            pic_path = str(queryset.filter(id=parameter).get().path)
            absolute_path = os.path.join(settings.MEDIA_ROOT, pic_path)
            os.remove(absolute_path)
        else:
            category_path = os.path.join(settings.MEDIA_ROOT, 'upload_images', str(user_id), parameter)
            shutil.rmtree(category_path)

        response = super(PicDeleteView, self).delete(request, *args, **kwargs)
        return response
