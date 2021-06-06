import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from matplotlib import pyplot as plt
import numpy as np
import os
from tensorflow.keras.layers.experimental import preprocessing
import seaborn as sns
from tensorflow.keras.callbacks import ReduceLROnPlateau

print(tf.__version__)

# Definir rutas datos
path_data = "data/"
path_data_training = path_data + "nsynth_train/audio/"
path_data_test = path_data + "nsynth_test/audio/"
path_data_valid = path_data + "nsynth_valid/audio/"

note_names = ['A0','A#0','B0','C1','C#1','D1','D#1','E1','F1','F#1','G1','G#1','A1','A#1','B1',
'C2','C#2','D2','D#2','E2','F2','F#2','G2','G#2','A2','A#2','B2',
'C3','C#3','D3','D#3','E3','F3','F#3','G3','G#3','A3','A#3','B3',
'C4','C#4','D4','D#4','E4','F4','F#4','G4','G#4','A4','A#4','B4',
'C5','C#5','D5','D#5','E5','F5','F#5','G5','G#5','A5','A#5','B5',
'C6','C#6','D6','D#6','E6','F6','F#6','G6','G#6','A6','A#6','B6',
'C7','C#7','D7','D#7','E7','F7','F#7','G7','G#7','A7','A#7','B7',
'C8']

# Preprocesamiento de los datos
# - Carga
# - Conversion a espectograma

def get_waveform_and_label(file_path):
  parts = tf.strings.split(file_path, "-")
  label = int(parts[1]) - 21
  audio_binary = tf.io.read_file(file_path)
  audio, _ = tf.audio.decode_wav(audio_binary)
  waveform = tf.squeeze(audio, axis=1)
  return waveform, label

def get_spectrogram(waveform):
  zero_padding = tf.zeros([64001] - tf.shape(waveform), dtype=tf.float32)
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

def filter_invalid_pitch(unString):
  return int(unString.split("-")[1]) >= 21

def get_data(path_data):
  train_files = list(filter(filter_invalid_pitch,list(map(lambda path: path_data + path, os.listdir(path_data)))))
  files_ds = tf.data.Dataset.from_tensor_slices(train_files)
  waveform_ds = files_ds.map(get_waveform_and_label)
  return waveform_ds.map(get_spectrogram_and_label_id)

# Levantar los datos de entrada (entrenamiento, test y validacion)
#  - Entrada: Archivo .WAV [-1, 1.0 ,0.5]

ds_train = get_data(path_data_training)
ds_valid = get_data(path_data_valid)
ds_test = get_data(path_data_test)

# Crear batchs de los datos

batch_size = 64
ds_train = ds_train.batch(batch_size)
ds_valid = ds_valid.batch(batch_size)

# Optimizacion para el entrenamiento

ds_train = ds_train.cache().prefetch(tf.data.AUTOTUNE)
ds_valid = ds_valid.cache().prefetch(tf.data.AUTOTUNE)

# Creacion del modelo de la red neuronal

for spectrogram, _ in ds_train.take(1):
  input_shape = spectrogram.shape
  
print('Input shape:', input_shape)
num_labels = 109

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
dot_img_file = '/tmp/model_1.png'

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=True, show_dtype=True,
    show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96
)

# Entrenamiento

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_lr=0.0001, verbose=1)
EPOCHS = 10
history = model.fit(
    ds_train,
    validation_data=ds_valid,
    batch_size=64,
    shuffle = True,
    epochs=EPOCHS,
    verbose='auto',
    callbacks=[reduce_lr] 
)

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.savefig("loss")

# Tests de precision

test_audio = []
test_labels = []

for audio, label in ds_test:
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)


y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')

# Matriz de confusion

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 

figure = plt.figure()
axes = figure.add_subplot(111)
caxes = axes.matshow(confusion_mtx, interpolation ='nearest')
figure.colorbar(caxes)
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.savefig("colorcitos")
plt.show()