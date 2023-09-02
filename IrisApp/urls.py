from django.urls import path, include
from . import views
from .views import *

app_name = 'IrisApp'

urlpatterns = [
    path('', views.home, name='home'),
    path('result', views.view_results, name="results"),
    
    path('home', views.home, name='home'),
    
    path('heartdiseaseanalyze', views.heartdiseaseanalyze, name='heartdiseaseanalyze'),
    path('diabeteseanalyze', views.diabeteseanalyze, name='diabeteseanalyze'),
    
    path('maleria', views.maleria, name='maleria'),
    
    # path('login/', authview.loginpage, name="loginpage"),
    # path('results/', views.view_results, name='results'),
    # path('predict/', views.predict_changes, name='submit_prediction'),
    
    
]