from django.shortcuts import render
from django.views.generic.edit import FormView
from django.urls import reverse_lazy

from .forms import PicProcessingForm

from picture.models import Picture

def pic_processing(request):
    if request.method == 'POST':
        form = PicProcessingForm(request.POST)
        print('success')
        # success_url = reverse_lazy('alogrithm:pic_pro_visualization')
        return render(request, 'algorithm/pic_pro_visualization', {'form': form})
    else:
        form = PicProcessingForm()

    return render(request, 'algorithm/pic_processing.html', {'form': form})



# class pic_processing(FormView):
#     form_class = PicProcessingForm
#     template_name = 'algorithm/pic_processing.html'
#     # success_url = reverse_lazy('alogrithm:pic_pro_visualization')


def pic_pro_visualization(request):
    pic_size = request.POST.pic_size
    test_pic = request.POST.test_pic
    orientations = request.POST.orientations
    pixels_per_cell = request.POST.pixels_per_cell
    cells_per_block = request.POST.cells_per_block
    is_color = request.POST.is_color

    form = PicProcessingForm()
    return render(request, 'algorithm/pic_processing.html', {'form': form})
