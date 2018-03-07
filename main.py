from lecturaFichero import lecturaFichero
from preparacionDatos import preparacionDatos
from generacionModelo import generacionModelo
from resultadosComparacion import resultadosComparacion
from puestaProduccion import puestaProduccion

def main ():
    data = lecturaFichero ("C:/Users/Juancarlos.martinezR/PycharmProjects/text_classification/data/yelp_academic_dataset_review_.json")
    X_train, X_test, y_train, y_test = preparacionDatos(data)
    #modelLinearSVC, modelDTree, modelExtraTrees, modelSGD, modelNB, modelMLP, modelGridSearch
    modelLinearSVC, modelMLP = generacionModelo(X_train, y_train)
    resultadosComparacion(modelLinearSVC, modelMLP, X_test, y_test) #modelLinearSVC, modelDTree, modelExtraTrees, modelSGD, modelNB, modelMLP, modelGridSearch, X_test, y_test)
    puestaProduccion(modelLinearSVC, modelMLP)  #modelLinearSVC, modelDTree, modelExtraTrees, modelSGD, modelNB)

if __name__== "__main__":
    main()

main()