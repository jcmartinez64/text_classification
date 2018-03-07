from django.conf.urls import url
from prodModelo import views

urlpatterns = [
    url(r'^$', views.bienvenida),
    url(r'^frase$', views.frase_list, name='frase'),
   # url(r'^new/(?P<pk>[0-9]+)/$', views.post_detail, name='post_detail'),
    #url(r'^new/sentence/$', views.new_sentence, name='new_sentence'),
]