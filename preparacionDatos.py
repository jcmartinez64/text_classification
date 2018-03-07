import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split

#### PREPARACIÓN DE LOS DATOS
def preparacionDatos (datos):
    # Cargamos las variables stemmer y stop_words que contendran los lexemas de las palabras y palabras no útiles, respectivamente.
    stemmer = SnowballStemmer('english', ignore_stopwords=True)
    stop_words = stopwords.words('english')

    # Creamos una columna nueva, "cleaned_text", que se quedará con los lexemas del stemmer, eliminará palabras no útiles de stop_words y pasará el texto a minúscula.
    datos['cleaned_text'] = datos['text'].apply(
    lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stop_words]).lower())

    # Generamos los sets de entreno y validación con una proporción 80-20
    X_train, X_test, y_train, y_test = train_test_split(datos['cleaned_text'], datos.stars, test_size=0.2)
    return (X_train, X_test, y_train, y_test)