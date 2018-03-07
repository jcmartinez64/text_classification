import json as j
import pandas as pd


### LECTURA DE FICHERO
#  Leemos el fichero fuente json, lo parseamos y lo cargamos en un dataframe de pandas
def lecturaFichero (ruta):
    json_data = None
    with open(ruta) as data_file:
        lines = data_file.readlines()
        joined_lines = "[" + ",".join(lines) + "]"
        json_data = j.loads(joined_lines)
        dataframe = pd.DataFrame(json_data)
        print ("Estructura de los datos:")
        dataframe.info()
    return dataframe