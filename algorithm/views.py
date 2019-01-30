from django.shortcuts import render
from django.views.generic.edit import FormView
from django.urls import reverse_lazy

from .forms import PicProcessingForm

from picture.models import Picture
from skimage import feature, exposure
from django.http import HttpResponse
import io
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2


def pic_processing(request):
    # if request.method == 'POST':
    #     form = PicProcessingForm(request.POST)
    #     print('success')
    #     # success_url = reverse_lazy('alogrithm:pic_pro_visualization')
    #     return render(request, 'algorithm/pic_pro_visualization/', {'form': form})
    # else:
    #     form = PicProcessingForm()
    #
    # return render(request, 'algorithm/pic_processing.html', {'form': form})
    return get_svg(request)




def setPlt(orientations, pixels_per_cell, cells_per_block, is_color=True):
    # numPts = 50
    # x = [random.random() for n in range(numPts)]
    # y = [random.random() for n in range(numPts)]
    # sz = 2 ** (10*np.random.rand(numPts))
    # plt.scatter(x, y, s=sz, alpha=0.5)
    pic_path = '/Users/sweeney/Workspaces/Graduate_Design/Test/picture/cat/1.jpg'
    pic_name = pic_path.split('/')[-1]
    img = cv2.imread(pic_path, 1)
    # 将opencv的BGR模式转为matplotlib的RGB模式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fd, hog_image = feature.hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                cells_per_block=cells_per_block, block_norm='L2-Hys', visualize=True,
                                multichannel=is_color)

    # print(hog_image.shape, type(hog_image))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('Input image: %s' % pic_name)

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    # plt.show()

def pltToSvg():
    buf = io.BytesIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    s = buf.getvalue()
    buf.close()
    return s

def get_svg(request):
    setPlt(9, (8, 8), (3, 3), is_color=True) # create the plot
    svg = pltToSvg() # convert plot to SVG
    plt.cla() # clean up plt so it can be re-used
    response = HttpResponse(svg, content_type='image/svg+xml')
    return response

def pic_pro_visualization(request):
    # pic_size = request.POST.pic_size
    # test_pic = request.POST.test_pic
    # orientations = request.POST.orientations
    # pixels_per_cell = request.POST.pixels_per_cell
    # cells_per_block = request.POST.cells_per_block
    # is_color = request.POST.is_color
    #
    # form = PicProcessingForm()
    # return render(request, 'algorithm/pic_processing.html', {'form': form})

    return get_svg(request)