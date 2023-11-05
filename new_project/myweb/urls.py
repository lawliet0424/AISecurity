"""myweb URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from protectphoto import views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.upload_file, name='upload_file'),
    # path('images/', views.result_list, name='result_list'),
    path('results/', views.result_list, name='result_list'),
    path('download_and_delete_image/<int:pk>/', views.download_and_delete_image, name='download_and_delete_image'),
    
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
