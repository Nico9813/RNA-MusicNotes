import tensorflow as tf

print(tf.__version__)

# Definir parametros globales 
rna_cant_neuronas_capas_ocultas = '800, 400, 200' #@param {type:"string"}
rna_tipo_capa_salida = 'softmax-MultiClase' #@param ["lineal-Numero", "softmax-MultiClase"] (???????)
rna_cant_epocas_entrenamiento =  200#@param {type:"integer"}
rna_learning_rate_entrenamiento = 0.001 #@param {type:"number"}
cantEpocas = (100 if rna_cant_epocas_entrenamiento<1 else rna_cant_epocas_entrenamiento)

# Levantar los datos de entrada (entrenamiento, test y validacion)
#  - Entrada: Archivo .WAV [-1, 1.0 ,0.5]

# Normalizar el tamaño de los datos de entrada
# Generar diccionario auxiliar para poder convertir de ID de clase a nombre de clase

# Crear el modelo 
# - Capa de entrada
# - Capas ocultas
# - Capa de salida

