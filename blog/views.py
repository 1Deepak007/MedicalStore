from django.shortcuts import render
from .models import Blogpost
from django.contrib import messages
from django.http import JsonResponse,HttpRequest

# Create your views here.
# def index(request):
#     return render(request, "blog/index.html")


def index(request):
    myposts = Blogpost.objects.all()
    print(myposts)
    return render(request,'blog/index.html',  {'myposts':myposts})

def blogpost(request, id):
    post = Blogpost.objects.filter(post_id = id)[0]
    print(post)
    return render(request, 'blog/blogpost.html', {'post':post})