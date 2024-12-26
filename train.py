import pathlib
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
tf.get_logger().setLevel('NOTSET')

from funcs import load_dataset


dataset, questions, answers = load_dataset()

print(f"Questions-answers gotten: {len(dataset)}")

rubert_model = SentenceTransformer("data/rubert_base")

print("Questions array embedding")
question_emb = rubert_model.encode(questions)
print("Answers array embedding")
answer_emb = rubert_model.encode(answers)

X, y = [], []
for i, q_emb in enumerate(question_emb):
    for j, a_emb in enumerate(answer_emb):
        X += [np.concatenate([q_emb, a_emb])]
        y += [int(i == j)]

X, y = np.array(X), np.array(y)

print(f"X shape: {X.shape}, y shape: {y.shape}")

model = tf.keras.models.Sequential()
inp_shape = X.shape[-1]

model.add(tf.keras.layers.InputLayer(input_shape=(inp_shape,)))
model.add(tf.keras.layers.Dense(200, activation='selu'))
model.add(tf.keras.layers.Dense(100, activation='selu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.AUC(curve='pr', name='auc')])

model.summary()

model.fit(X, y,
          epochs=1000
)

path = pathlib.Path(__file__).parent / "data"
model.save(path / "model.keras")
