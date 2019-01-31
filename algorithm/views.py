from django.shortcuts import render
from django.conf import settings

from .forms import PicProcessingForm

from os import path
from skimage import feature, exposure
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def hog_pic(request):

    if request.method == 'POST':

        # get the parameter
        test_pic = request.FILES.get('test_pic')
        pic_size = request.POST.get('pic_size')
        orientations = int(request.POST.get('orientations'))
        pixels_per_cell = request.POST.get('pixels_per_cell')
        cells_per_block = request.POST.get('cells_per_block')
        is_color = request.POST.get('is_color')

        str_to_tuple = lambda a_str: tuple((int(a_str.split(',')[0]), int(a_str.split(',')[1])))

        # format the parameter
        pic_size = str_to_tuple(pic_size)
        pixels_per_cell = str_to_tuple(pixels_per_cell)
        cells_per_block =str_to_tuple(cells_per_block)

        #  save the test_pic
        pic_name = request.FILES.get('test_pic').name
        ext = pic_name.split('.')[-1]
        saved_pic_path = path.join(settings.MEDIA_ROOT, 'algorithm', 'hog_picture',  str(request.user.id) + '_' + 'hog_test_pic' + '.' + ext)
        with open(saved_pic_path, 'wb+') as destination:
            for chunk in test_pic.chunks():
                destination.write(chunk)

        # read pic and resize it
        img = cv2.imread(saved_pic_path, 1)
        img = cv2.resize(img, pic_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(saved_pic_path, img)

        # 将opencv的BGR模式转为matplotlib的RGB模式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # use the hog to change the picture
        fd, hog_image = feature.hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                    cells_per_block=cells_per_block, multichannel=is_color,
                                    block_norm='L2-Hys', visualize=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        # show the origin picture
        ax1.axis('off')
        ax1.imshow(img, cmap=plt.cm.gray)
        ax1.set_title('Input image: %s' % pic_name)

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        # show hoged picture
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')

        # save the plt as png
        hog_picture_path = path.join(settings.MEDIA_ROOT, 'algorithm', 'hog_picture',  str(request.user.id) + '_hog_picture' + '.' + ext)
        plt.savefig(hog_picture_path)

        # get the saved png to show in page
        relative_pic_path = hog_picture_path.replace(settings.BASE_DIR, '')
        form = PicProcessingForm(request.POST)
        return render(request, 'algorithm/pic_processing.html', {'form': form, 'hog_picture': relative_pic_path})

    else:
        form = PicProcessingForm()
        return render(request, 'algorithm/pic_processing.html', {'form': form})


def choose_pic_category(request):
    pass

