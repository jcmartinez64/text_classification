import numpy as np
import random
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from time import time
import sys


### GENERACION DEL MODELO
# Aplicamos las funciones de vectorización, selección de features de acuerdo a los k scores más altos y configuramos el clasificador que usaremos
def generacionModelo (X_train, y_train):
    seed = 8
    random.seed(seed)

    #### SVM - LinearSVC ####
    #Pipeline para SVM - LinearSVC
    pipeline1 = Pipeline([('vector', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                        ('chi2',  SelectKBest(chi2, k=3000)),
                        ('class', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))]) # Penalty = I1 --> tratar vectores de coefientes dispersos y dual=False porque n_samples > n_features.

    # Ejecutamos el modelo SVM - LinearSVC

    log("")
    log("===============================================")
    log("Ejecutando modelo SVM - LinearSVC")
    now = time()
    modelLinearSVC = pipeline1.fit(X_train, y_train)
    log("Tiempo de entrenamiento: {0}s".format(time() - now))
    log("===============================================")


    # ### Arbol de decision ####
    # #Pipeline para Arbol de decision
    # pipeline2 = Pipeline([('vector', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
    #                  ('chi2',  SelectKBest(chi2, k=3000)),
    #                  ('class', DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=seed, presort=False))])
    #
    # # Ejecutamos el modelo Arbol de decision
    # log("")
    # log("===============================================")
    # log("Ejecutando modelo Arbol de decision")
    # now = time()
    # modelDTree = pipeline2.fit(X_train, y_train)
    # log("Tiempo de entrenamiento: {0}s".format(time() - now))
    # log("===============================================")
    #
    # ### ExtraTrees ####
    # #Pipeline para Word2vec
    # pipeline3 = Pipeline([('vector', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
    #                  ('chi2',  SelectKBest(chi2, k=3000)),
    #                  ('class', ExtraTreesClassifier(n_estimators=10))])
    #
    # # Ejecutamos el modelo ExtraTrees
    # log("")
    # log("===============================================")
    # log("Ejecutando modelo Word2Vec")
    # now = time()
    # modelExtraTrees = pipeline3.fit(X_train, y_train)
    # log("Tiempo de entrenamiento: {0}s".format(time() - now))
    # log("===============================================")


    # #### SVM SGD ####
    # #Pipeline para SVM SGD
    # pipeline4 = Pipeline([('vector', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
    #                  ('chi2',  SelectKBest(chi2, k=3000)),
    #                  ('class', SGDClassifier(loss='hinge', penalty='l1', alpha=1e-3, max_iter=5, random_state=seed))])
    #
    # #Ejecutamos el modelo SVM SGD
    # log("")
    # log("===============================================")
    # log("Ejecutando modelo SVM SGD")
    # now = time()
    # modelSGD = pipeline4.fit(X_train, y_train)
    # log("Tiempo de entrenamiento: {0}s".format(time() - now))
    # log("===============================================")

    # #### Naive Bayes ####
    # #Pipeline para Naive Bayes
    # pipeline5 = Pipeline([('vector', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
    #                  ('chi2',  SelectKBest(chi2, k=3000)),
    #                  ('class', MultinomialNB())])
    # # Ejecutamos el modelo Naive Bayes
    # log("")
    # log("===============================================")
    # log("Ejecutando modelo Naive Bayes")
    # now = time()
    # modelNB = pipeline5.fit(X_train, y_train)
    # log("Tiempo de entrenamiento: {0}s".format(time() - now))
    # log("===============================================")

    #### GRIDSEARCH ####
    # Utilizamos GridSearch para optimizar el resultado del modelo
    # log("")
    # log("================================================================")
    # log("Ejecutando GridSearchCV con clasificador SGD")

    # #Pipeline para GridSearch
    # pipeline6 = Pipeline([('vector', TfidfVectorizer()),
    #                     ('class', SGDClassifier(loss='hinge', penalty='l2', random_state=seed))])
    #
    # parameters = {
    #     'vector__max_df': (0.5, 0.75, 1.0),   #, 0.75, 1.0),
    #     # 'vector__max_features': (None, 5000, 10000, 50000),
    #     'vector__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #     # 'tfidf__use_idf': (True, False),
    #     # 'tfidf__norm': ('l1', 'l2'),
    #     'class__alpha': (0.0001, 0.001),
    #     #'class__penalty': ('l2', 'elasticnet'),
    #     # 'class__n_iter': (10, 50, 80),
    # }
    #
    # grid_search = GridSearchCV(pipeline6, parameters)
    #
    # print (grid_search)
    #
    # print("Performing grid search...")
    # print("Pipeline:", [name for name, __ in pipeline6.steps])
    # print("Parameters:")
    # print(parameters)
    # now = time()
    #
    # try:
    #     modelGridSearch = grid_search.fit(X_train, y_train)
    # except:
    #     print("Error")
    #     sys.exit(1)
    #
    # log("Tiempo de entrenamiento: {0}s".format(time() - now))
    # print("Mejor score: %0.3f" % grid_search.best_score_)
    # print("Mejores parametros:")
    # best_parameters = grid_search.best_estimator_.get_params()
    # for param_name in sorted(parameters.keys()):
    #     print("\t%s: %r" % (param_name, best_parameters[param_name]))
    # log("===============================================")


    #### MLP ####
    #Pipeline para MLP
    pipeline7 = Pipeline([('vector', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                     ('chi2',  SelectKBest(chi2, k=3000)),
                     ('class', MLPClassifier(hidden_layer_sizes=(20, 3), max_iter=150, alpha=1e-4, solver='sgd', verbose=10, tol=1e-4, random_state=seed, learning_rate_init=.1))])
    # Ejecutamos el modelo MLP
    log("")
    log("===============================================")
    log("Ejecutando modelo MLP")
    now = time()
    modelMLP = pipeline7.fit(X_train, y_train)
    log("Tiempo de entrenamiento: {0}s".format(time() - now))
    log("===============================================")

    #Persistemos el modelo en un archivo para rescatarlo en Pathos (aplicación web hecha en Django)
    file_Name_SVC = "fileSVC"
    file_Name_MLP = "fileMLP"
    # open the file for writing
    fileObjectSVC = open(file_Name_SVC, 'wb')
    fileObjectMLP = open(file_Name_MLP, 'wb')
    pickle.dump(modelLinearSVC, fileObjectSVC)
    pickle.dump(modelMLP, fileObjectMLP)
    fileObjectSVC.close()
    fileObjectMLP.close()


    return (modelLinearSVC, modelMLP)           #modelLinearSVC, modelDTree, modelExtraTrees, modelSGD, modelNB, modelGridSearch, modelMLP)

def log(x):
    #can be used to write to log file
    print(x)