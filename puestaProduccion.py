
def puestaProduccion (modelLinearSVC, modelDTree, modelExtraTrees, modelSGD, modelNB, modelMLP): #modelLinearSVC, modelDTree, modelExtraTrees, modelSGD, modelNB):
    frase = 'Incredible food! Amazing place'
    print("La frase propuesta <"+ frase +"> obtiene un rating de:")
    print("Según el modelo modelLinearSVC:")
    print(modelLinearSVC.predict([frase]))
    print("Según el modelo modelDTree:")
    print(modelDTree.predict([frase]))
    print("Según el modelo modelExtraTrees:")
    print(modelExtraTrees.predict([frase]))
    print("Según el modelo modelSGD:")
    print(modelSGD.predict([frase]))
    print("Según el modelo modelMLP:")
    print(modelMLP.predict([frase]))