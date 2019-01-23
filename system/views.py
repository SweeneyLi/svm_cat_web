from django.shortcuts import render
from django.http import HttpResponseRedirect
# Create your views here.


def index(request):

    if request.user.is_authenticated:
        user = request.user.username
    else:
        user = None

    return render(request, 'system/index.html', {'user': user})

