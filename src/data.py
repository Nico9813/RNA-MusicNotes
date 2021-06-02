import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print(tf.__version__)

# Definir rutas datos
path_data = "data"
path_data_training = path_data + "/nsynth-test.tfrecord"

# Definir parametros globales 
rna_cant_neuronas_capas_ocultas = '800, 400, 200' #@param {type:"string"}
rna_tipo_capa_salida = 'softmax-MultiClase' #@param ["lineal-Numero", "softmax-MultiClase"] (???????)
rna_cant_epocas_entrenamiento =  200#@param {type:"integer"}
rna_learning_rate_entrenamiento = 0.001 #@param {type:"number"}
cantEpocas = (100 if rna_cant_epocas_entrenamiento<1 else rna_cant_epocas_entrenamiento)

# Levantar los datos de entrada (entrenamiento, test y validacion)
#  - Entrada: Archivo .WAV [-1, 1.0 ,0.5]

def get_data(file):
    dataset = tf.data.TFRecordDataset([file])

    x = []
    y = []

    i = 0

    for element in dataset:
        if i < 100:
            record = tf.train.Example()
            record.ParseFromString(element.numpy())

            record_input = record.features.feature["audio"].float_list.value
            record_label = record.features.feature["pitch"].int64_list.value[0]

            x.append(record_input)
            y.append(record_label)
            
            i += 1
    return np.array(x),np.array(y)
    
x_train, y_train = get_data(path_data_training)

# x_test, y_test = get_data(path_data_test)
# x_validation, y_validation = get_data(path_data_validation)

# Generar graficos de las entradas

# Normalizar el tamaño de los datos de entrada
# Generar diccionario auxiliar para poder convertir de ID de clase a nombre de clase

# Crear el modelo
# - Capa de entrada

input_layer = layers.Dense(2, activation="relu", name="input_raw_file")
# hidden_layers = []
output_layer = layers.Dense(4, name="output_pitch")

model = keras.Sequential([
    input_layer,
    output_layer
])

opt = keras.optimizers.Adam(learning_rate=rna_learning_rate_entrenamiento)

model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=cantEpocas)

model.summary()

# - Capas ocultas
# - Capa de salida


