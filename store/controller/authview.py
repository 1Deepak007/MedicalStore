from django.shortcuts import render,redirect,HttpResponse
from django.contrib import messages
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.models import User
from store.forms import CustomUserForm


def register(request):
    form = CustomUserForm()
    if request.method == 'POST':
        form = CustomUserForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request,"Registered successfully !   Login to continue....")
            return redirect('/login')
    context = {'form': form}
    return render(request, "store/auth/register.html",context)


def loginpage(request):
    if request.user.is_authenticated:
        messages.warning(request, "You are already logged in")
        return redirect('/')
    else:
        if request.method == "POST":
            name = request.POST.get('username')
            passwd = request.POST.get('password')
        
            user = authenticate(request, username=name, password=passwd)
        
            if user is not None:
                login(request, user)
                messages.success(request, "LogIn Susccessfull")
                return redirect("/")
            else:
                messages.error(request, "Invalid username or password")
                return redirect("/login")        
        
        return render(request, "store/auth/login.html")
    
def logoutpage(request):
    if request.user.is_authenticated:
        logout(request)
        messages.success(request,"Logged out successfully")
        return redirect("/")
    
    