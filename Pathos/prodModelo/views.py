from django.shortcuts import render, render_to_response
from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.template import Context, loader
from .forms import FraseForm
import os
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pandas as pd

def bienvenida (request):
    modulePath = os.path.dirname(__file__)  # get current directory
    filePathSVC = os.path.join(modulePath, 'fileSVC')
    filePathMLP = os.path.join(modulePath, 'fileMLP')
    global modelSVC
    global modelMLP
    with open(filePathSVC, 'rb') as f1:
        modelSVC = pickle.load(f1)
    with open(filePathMLP, 'rb') as f2:
        modelMLP = pickle.load(f2)
        return render_to_response('home.html')


def frase_list(request):
    form = FraseForm(request.POST)
    global sentence
    global sentence_clean
    global ratioSVC
    global ratioMLP
    global centinela
    #global context
    #global resultado
    sentence = ""
    ratioSVC = 0
    ratioMLP = 0
    centinela = False
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # Cargamos las variables stemmer y stop_words que contendran los lexemas de las palabras y palabras no útiles, respectivamente.
            centinela = True
            stemmer = SnowballStemmer('english', ignore_stopwords=True)
            stop_words = stopwords.words('english')
            sentence = (request.POST.get('frase'))
            print(sentence)
            sentence = sentence.lower()
            print(sentence)
            sentence_clean = pd.DataFrame({'sentence': [sentence]})

            # Creamos una columna nueva, "sentence_clean", que se quedará con los lexemas del stemmer, eliminará palabras no útiles de stop_words y pasará el texto a minúscula.

            sentence_clean['sentence_clean'] = sentence_clean['sentence'].apply(
                lambda x: " ".join(
                    [stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stop_words]))

            sentence = sentence_clean[['sentence_clean']].to_string(index=False, header=None)
            print(type(sentence))
            print(sentence)
            ratioMLP = modelMLP.predict([sentence])
            ratioSVC = modelSVC.predict([sentence])
            print(ratioSVC)
            print(ratioMLP)

            # redirect to a new URL:
            #return HttpResponseRedirect('frase', sentence)

    # if a GET (or any other method) we'll create a blank form
    else:
        centinela = False
        form = FraseForm()

    #resultado = ({"form": form, "sentence": sentence, "ratioMLP": ratioMLP, "ratioSVC": ratioSVC})
    #context = Context({'resultado':resultado})
    return render(request,'frase_list.html',{"form": form, "sentence": sentence, "ratioMLP": ratioMLP, "ratioSVC": ratioSVC})
