from django.contrib import admin
from .models import *
from django.contrib.auth.models import Group
from django.db.models import Count

#=========> Register your models here.

admin.site.unregister(Group)

# admin.site.register(Order)
admin.site.register(OrderItem)
admin.site.register(Profile)
admin.site.register(Category)
#============================================================================

class ProductDtls(admin.ModelAdmin):
    list_display = ['name','quantity','expiry_date','category']
admin.site.register(Product,ProductDtls)


class ExpMeds(admin.ModelAdmin):
    list_display = ['name','quantity','expiry_date']
admin.site.register(ExpiredMedecines,ExpMeds)

class Ordr(admin.ModelAdmin):
    list_display = ['tracking_no','status']
admin.site.register(Order,Ordr)
# admin.site.register(Order)



# def count_products_per_category():
#     categories_with_counts = Category.objects.annotate(product_count=Count('product'))
#     for category in categories_with_counts:
#         print(f"{category.name}: {category.product_count} products")
    


# class Ctgry(admin.ModelAdmin):
#     categories_with_counts = Category.objects.annotate(product_count=Count('product'))
#     for category in categories_with_counts:
#         print(f"{category.name}: {category.product_count} products")

#     list_display = ['Total Products']
# admin.site.register(Category, Ctgry)
