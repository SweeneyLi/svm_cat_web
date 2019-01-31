from django.shortcuts import render
from django.views.generic.edit import FormView
from django.urls import reverse_lazy
from django.http import HttpResponse
from django.shortcuts import redirect

from .forms import PicProcessingForm
from picture.models import Picture

from skimage import feature, exposure

import cv2
import io
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



# # input: '8,8'. output:(8,8)
# def str_to_tuple(a_str):
#


def pic_processing(request):
    if request.method == 'POST':
        # pic_size = request.form.pic_sie
        # test_pic = request.form.test_pic
        orientations = int(request.POST.get('orientations'))
        pixels_per_cell = request.POST.get('pixels_per_cell')
        cells_per_block = request.POST.get('cells_per_block')
        is_color = request.POST.get('is_color')

        str_to_tuple = lambda a_str: tuple((int(a_str.split(',')[0]), int(a_str.split(',')[1])))

        pixels_per_cell = str_to_tuple(pixels_per_cell)
        cells_per_block =str_to_tuple(cells_per_block)

        # setplt
        pic_path = '/Users/sweeney/Workspaces/Graduate_Design/Test/picture/cat/1.jpg'
        pic_name = pic_path.split('/')[-1]
        img = cv2.imread(pic_path, 1)

        # 将opencv的BGR模式转为matplotlib的RGB模式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fd, hog_image = feature.hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                    cells_per_block=cells_per_block, multichannel=is_color,
                                    block_norm='L2-Hys', visualize=True)

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

        plt.savefig('temp.png')
        # convert plot to SVG
        svg = pltToSvg()
        plt.cla()  # clean up plt so it can be re-used
        response = HttpResponse(svg, content_type='image/svg+xml')
        return response

    else:
        form = PicProcessingForm()

    return render(request, 'algorithm/pic_processing.html', {'form': form})


def pltToSvg():
    buf = io.BytesIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    s = buf.getvalue()
    buf.close()
    return s

