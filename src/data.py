import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from matplotlib import pyplot as plt
import numpy as np
import os
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)

# Definir rutas datos
path_data = "data/"
path_data_training = path_data + "nsynth_test/audio/"
path_data_test = path_data + "nsynth_test/audio/"
path_labels_training = path_data + "nsynth_test/examples.json"

# Levantar los datos de entrada (entrenamiento, test y validacion)
#  - Entrada: Archivo .WAV [-1, 1.0 ,0.5]

def get_waveform_and_label(file_path):
  parts = tf.strings.split(file_path, "-")
  label = int(parts[1])
  audio_binary = tf.io.read_file(file_path)
  audio, _ = tf.audio.decode_wav(audio_binary)
  waveform = tf.squeeze(audio, axis=1)
  return waveform, label

def get_spectrogram(waveform):

  # Concatenate audio with padding so that all audio clips will be of the
  # same length
  zero_padding = tf.zeros([64001] - tf.shape(waveform), dtype=tf.float32)
  # Concatenate audio with padding so that all audio clips will be of the
  # same length
  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)

  spectrogram = tf.abs(spectrogram)

  print("waveform " + str(waveform.shape))
  print("esepctograma " + str(spectrogram.shape))
  return spectrogram

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  spectrogram = tf.expand_dims(spectrogram, -1)
  label_id = label
  return spectrogram, label_id

def get_data(path_data):
  train_files = list(map(lambda path: path_data + path, os.listdir(path_data)))
  files_ds = tf.data.Dataset.from_tensor_slices(train_files)
  #for element in files_ds.as_numpy_iterator():
  #  print(element)
  waveform_ds = files_ds.map(get_waveform_and_label)
  return waveform_ds.map(get_spectrogram_and_label_id)

ds_train = get_data(path_data_training)
ds_test = get_data(path_data_test)

batch_size = 64
ds_train = ds_train.batch(batch_size)
ds_test = ds_test.batch(batch_size)

ds_train = ds_train.cache().prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.cache().prefetch(tf.data.AUTOTUNE)

for spectrogram, _ in ds_train.take(1):
  input_shape = spectrogram.shape
  
print('Input shape:', input_shape)
num_labels = 150

norm_layer = preprocessing.Normalization()
norm_layer.adapt(ds_train.map(lambda x, _: x))

model = models.Sequential([
    layers.InputLayer(input_shape=(499, 129, 1)),
    preprocessing.Resizing(32, 32),
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 10
history = model.fit(
    ds_train,
    validation_data=ds_train,
    batch_size=64,
    epochs=EPOCHS,
    verbose='auto',
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.savefig("loss")




# x_test, y_test = get_data(path_data_test)
# x_validation, y_validation = get_data(path_data_validation)

# Generar graficos de las entradas

# Normalizar el tamaÃ±o de los datos de entrada
# Generar diccionario auxiliar para poder convertir de ID de clase a nombre de clase

