from django.shortcuts import get_object_or_404, render,redirect
from .models import Product,ExpiredMedecines
from .models import *
from django.contrib import messages
from django.http import JsonResponse,HttpRequest
from datetime import date
import datetime
import pytz
# import requests
from itsdangerous import Serializer


def dashboard_with_pivot(request):
    return render(request, 'store/dashboard_with_pivot.html', {})

    
def pivot_data(request):
    dataset = Order.objects.all()
    data = Serializer.serialize('json', dataset)
    return JsonResponse(data, safe=False)


# Create your views here.
def home(request):
    RmExpMediFrmProd()
    trending_products = Product.objects.filter(trending=1)
    context = {'trending_products':trending_products}
    
    return render(request, "store/index.html", context)

# Will display in collections.html
def collections(request):
    category = Category.objects.filter(status=0)
    context = {'category':category}
    return render(request ,"store/collections.html", context)


def collectionsview(request, slug):
    if(Category.objects.filter(slug=slug, status=0)):
        products = Product.objects.filter(category__slug=slug)
        category = Category.objects.filter(slug=slug).first()
        context = {'products':products, 'category':category}
        return render(request, "store/products/index.html",context)
    else:
        messages.warning(request, "No such category found")
        return redirect('collections')
    
def productview(request, cate_slug, prod_slug):
    if(Category.objects.filter(slug=cate_slug,status=0)):
        if(Product.objects.filter(slug=prod_slug,status=0)):
            products = Product.objects.filter(slug=prod_slug,status=0).first
            context = {'products': products}
        else:
            messages.error(request, "No such category found")
            return redirect('collections')    
    else:
        messages.error(request, "No such category found")
        return redirect('collections')
    return render(request,"store/products/view.html",context)


def productlistAjax(request):
    products = Product.objects.filter(status=0).values_list('name', flat=True)
    productList = list(products)
    
    return JsonResponse(productList, safe=False)     # changed, it was TRUE

def searchproduct(request):
    if request.method=='POST':
        searchedterm = request.POST.get('productsearch')
        if searchedterm == "":
            return redirect(request.META.get('HTTP_REFERER'))
        else:
            product = Product.objects.filter(name__contains = searchedterm).first()
            
            if product:
                return redirect('collections/'+product.category.slug+'/'+product.slug)
            else:
                messages.info(request, "No product matched your search")
                return redirect(request.META.get('HTTP_REFERER'))
    return redirect(request.Meta.get('HTTP_REFERER'))


# pop expired medecines from product table and add in ExpiredMedecines
def RmExpMediFrmProd():
    source_objects = Product.objects.all()
    for source_object in source_objects:
        target_object = ExpiredMedecines()
        
        exp_date_x = source_object.expiry_date
        now_date_x = datetime.datetime.now()
        
        
        exp_date = exp_date_x.replace(tzinfo=pytz.utc)
        now_date = now_date_x.replace(tzinfo=pytz.utc)
        
        print("==========================")
        # print(exp_date)
        # print(now_date)
        
        date_expiry = str(exp_date)
        print("expiry date : ",date_expiry[0:10])
        date_now = str(now_date)
        print("current date : ",date_now[0:10])
        
        print(date_now == date_expiry)
        print("==========================")
        
        pid = source_object.unique_key
        
        
        if date_now > date_expiry:
            if (source_object.name not in target_object.name and source_object.tag not in target_object.tag and source_object.meta_title not in target_object.meta_title):
                # Add in ExpiredMedecines
                target_object.slug = source_object.slug
                target_object.name =source_object.name
                target_object.product_image = source_object.product_image
                target_object.small_description = source_object.small_description
                target_object.quantity = source_object.quantity
                target_object.description = source_object.description
                target_object.original_price = source_object.original_price
                target_object.selling_price = source_object.selling_price
                target_object.tag = source_object.tag
                target_object.meta_title = source_object.meta_title
                target_object.meta_keywords = source_object.meta_keywords
                target_object.meta_description = source_object.meta_description
                target_object.created_at = source_object.created_at
                target_object.expiry_date = source_object.expiry_date
                target_object.save()
                
            #--> delete from product table
            # delete_product(source_object.unique_key)
                prod = get_object_or_404(Product, unique_key=source_object.unique_key)
                prod.delete()
            
        else:
            print("product not expired")


