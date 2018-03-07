# text_classification
Desarrollo y paso a producción de un sistema de clasificación de reseñas de YELP sobre restaurantes basado en NLP. Se ha desarrollado
siguiendo un diseño descendiente (top-down)
El código principal main () está compuesto de las siguientes funciones:
    
    data = lecturaFichero ("C:/Users/Juancarlos.martinezR/PycharmProjects/text_classification/data/yelp_academic_dataset_review_.json")
    X_train, X_test, y_train, y_test = preparacionDatos(data)
    #Modelos utilizados para la evaluación: modelLinearSVC, modelDTree, modelExtraTrees, modelSGD, modelNB, modelMLP, modelGridSearch
    modelLinearSVC, modelDTree, modelExtraTrees, modelSGD, modelNB, modelMLP, modelGridSearch = generacionModelo(X_train, y_train)
    resultadosComparacion(modelLinearSVC, modelDTree, modelExtraTrees, modelSGD, modelNB, modelMLP, modelGridSearch, X_test, y_test) 
    puestaProduccion(modelLinearSVC, modelDTree, modelExtraTrees, modelSGD, modelNB, modelMLP, modelGridSearch, X_test, y_test)

Entrando en cada uno de los scripts de las funciones se ve en detalle qué realiza cada uno.

#pathos
Es un desarrollo web basado en el framework Django. Nos permite poner en producción y usar los modelos entrenados con comentarios reales
de usuarios. Los evalua del 1 al 5 siendo 1 lo más negativo y 5 lo más positivo. 
