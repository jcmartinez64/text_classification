import numpy as np
from time import time
from sklearn.metrics import classification_report, confusion_matrix


def resultadosComparacion (modelLinearSVC, modelDTree, modelExtraTrees, modelSGD, modelNB, modelMLP, x_test, y_test):      #modelLinearSVC, modelDTree, modelExtraTrees, modelSGD, modelNB, x_test, y_test):

### Precisión del modelo LinearSVC
    print("Precisión del modelo LinearSVC: " + str(modelLinearSVC.score(x_test, y_test)))

    vectorizer1 = modelLinearSVC.named_steps['vector']
    chi1 = modelLinearSVC.named_steps['chi2']
    clf1 = modelLinearSVC.named_steps['class']

    feature_names = vectorizer1.get_feature_names()
    feature_names = [feature_names[i] for i in chi1.get_support(indices=True)]
    feature_names = np.asarray(feature_names)

    log("=================== Results ===================")
    # Palabras con mayor scoring para las cinco categorías de rating
    target_names = ['1', '2', '3', '4', '5']
    print("Las 10 palabras con mayor scoring para las cinco categorías de rating:")
    for i, label in enumerate(target_names):
         top10 = np.argsort(clf1.coef_[i])[-10:]
         print("%s: %s" % (label, " ".join(feature_names[top10])))
    now = time()
    predictions = modelLinearSVC.predict(x_test)
    log("Tiempo de predicción {0}s".format(time() - now))
    resultados_modelLinearSVC= classification_report(y_test, predictions, target_names=target_names)
    print(resultados_modelLinearSVC)
    print(confusion_matrix(y_test, predictions))
    log("===============================================")

#### Precisión del modelo DTree
    print("Precisión del modelo DTree: " + str( modelDTree.score(x_test, y_test)))
    predictions = modelDTree.predict(x_test)
    log("Tiempo de predicción {0}s".format(time() - now))
    resultados_modelDTree= classification_report(y_test, predictions, target_names=target_names)
    print(resultados_modelDTree)
    print(confusion_matrix(y_test, predictions))
    log("===============================================")

### Precisión del modelo ExtraTrees
    print("Precisión del modelo ExtraTrees: " + str(modelExtraTrees.score(x_test, y_test)))
    predictions = modelExtraTrees.predict(x_test)
    log("Tiempo de predicción {0}s".format(time() - now))
    resultados_modelExtraTrees= classification_report(y_test, predictions, target_names=target_names)
    print(confusion_matrix(y_test, predictions))
    log("===============================================")

#### Precisión del modelo SGD
    print("Precisión del modelo SVM-SGD: " + str(modelSGD.score(x_test, y_test)))

    vectorizer4 = modelSGD.named_steps['vector']
    chi4 = modelSGD.named_steps['chi2']
    clf4 = modelSGD.named_steps['class']

    feature_names = vectorizer4.get_feature_names()
    feature_names = [feature_names[i] for i in chi4.get_support(indices=True)]
    feature_names = np.asarray(feature_names)

    log("=================== Results ===================")
    # Palabras con mayor scoring para las cinco categorías de rating
    target_names = ['1', '2', '3', '4', '5']
    print("Las 10 palabras con mayor scoring para las cinco categorías de rating:")
    for i, label in enumerate(target_names):
         top10 = np.argsort(clf4.coef_[i])[-10:]
         print("%s: %s" % (label, " ".join(feature_names[top10])))

    now = time()
    predictions = modelSGD.predict(x_test)
    log("Tiempo de predicción {0}s".format(time() - now))
    resultados_modelSGD= classification_report(y_test, predictions, target_names=target_names)
    print(resultados_modelSGD)
    print(confusion_matrix(y_test, predictions))
    log("===============================================")


#### Precisión del modelo Naive Bayes
    print("Precisión del modelo Naive Bayes: " + str(modelNB.score(x_test, y_test)))

    vectorizer5 = modelNB.named_steps['vector']
    chi5 = modelNB.named_steps['chi2']
    clf5 = modelNB.named_steps['class']

    feature_names = vectorizer5.get_feature_names()
    feature_names = [feature_names[i] for i in chi5.get_support(indices=True)]
    feature_names = np.asarray(feature_names)

    log("=================== Results ===================")
    # Palabras con mayor scoring para las cinco categorías de rating
    target_names = ['1', '2', '3', '4', '5']
    print("Las 10 palabras con mayor scoring para las cinco categorías de rating:")
    for i, label in enumerate(target_names):
         top10 = np.argsort(clf5.coef_[i])[-10:]
         print("%s: %s" % (label, " ".join(feature_names[top10])))

    now = time()
    predictions = modelNB.predict(x_test)
    log("Tiempo de predicción {0}s".format(time() - now))
    resultados_modelNB= classification_report(y_test, predictions, target_names=target_names)
    print (confusion_matrix (y_test, predictions))
    print(resultados_modelNB)
    log("===============================================")


#### Precisión del modelo MLP
    print("Precisión del modelo modelo MLP: " + str(modelMLP.score(x_test, y_test)))
    predictions = modelMLP.predict(x_test)
    log("Tiempo de predicción {0}s".format(time() - now))
    resultados_modelMLP= classification_report(y_test, predictions, target_names=target_names)
    print(confusion_matrix(y_test, predictions))
    print(resultados_modelMLP)
    log("===============================================")

def log(x):
    #can be used to write to log file
    print(x)