from django.shortcuts import redirect, render
from django.contrib import messages

from store.models import Product,Cart
from django.contrib.auth.decorators import login_required

from store.models import Order, OrderItem

def index(request):
    orders = Order.objects.filter(user=request.user)
    context = {'orders':orders}
    return render(request, 'store/orders/index.html', context)
    

def vieworder(request, t_no):
    order = Order.objects.filter(tracking_no = t_no).filter(user = request.user).first()
    orderitems = OrderItem.objects.filter(order=order)
    context = {'order':order, 'orderitems':orderitems}
    return render(request, "store/orders/view.html",context)