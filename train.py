"""## Импортирование"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
# print(tf.__version__)

import warnings
warnings.filterwarnings("ignore")

import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
tf.get_logger().setLevel('NOTSET')

from funcs import load_dataset


bert_prepr = "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"
bert_model = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3"

"""## Препроцессор и модель BERT"""

print("BERT Preprocessor loading..")
bert_prepr = hub.KerasLayer(bert_prepr)
print("BERT Model loading..")
bert_model = hub.KerasLayer(bert_model)

"""## Считывание датасета"""

dataset, questions, answers = load_dataset()
print(f"Questions-answers gotten: {len(dataset)}")

"""## Эмбединг"""

data = []
for i, (question, answer) in enumerate(zip(questions, answers)):
    q_emb = bert_model(bert_prepr([question]))["pooled_output"].numpy()
    a_emb = bert_model(bert_prepr([answer]))["pooled_output"].numpy()

    data += [[np.array(q_emb[0]), np.array(a_emb[0])]]

    if i % len(dataset) // 10 == 0:
        print(i, end=' ')

data = np.array(data)
print(f"\nData shape: {data.shape}")

"""## Подготовка датасета"""

X, y = [], []
for i in range(data.shape[0]):
    for j in range(data.shape[0]):
        X += [np.concatenate([data[i, 0, :], data[j, 1, :]], axis=0)]
        y += [int(i == j)]

X = np.array(X)
y = np.array(y)

print(f"X shape: {X.shape}, y shape: {y.shape}")

"""## Модель чат-бота"""

model = tf.keras.models.Sequential()
inp_shape = X.shape[-1]

model.add(tf.keras.layers.InputLayer(input_shape=(inp_shape,)))
# model.add(tf.keras.layers.Dense(inp_shape // 2, activation='selu'))
# model.add(tf.keras.layers.Dense(inp_shape // 4, activation='selu'))
# model.add(tf.keras.layers.Dense(inp_shape // 8, activation='selu'))
model.add(tf.keras.layers.Dense(200, activation='selu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

es = tf.keras.callbacks.EarlyStopping(monitor='auc',
                                      mode='max',
                                      patience=10,
                                      restore_best_weights=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.AUC(curve='pr', name='auc')])

model.summary()

model.fit(X, y,
          epochs=1,
          # class_weight={0: 1, 1: len(dataset)}
)

path = pathlib.Path(__file__).parent / "data"
np.save(path / "data.npy", data)
model.save(path / "model_local.keras")
