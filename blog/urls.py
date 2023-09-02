from django.urls import path, include
from . import views

urlpatterns = [
    # path('admin/', admin.site.urls),
    # path(r'admin/', admin.site.urls),
    path('', views.index, name='BlogHome'),
    path('blogpost/<int:id>',views.blogpost, name='blogpost')
]