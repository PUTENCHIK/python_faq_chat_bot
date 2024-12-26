import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pathlib
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer

from funcs import load_dataset


bert_model = SentenceTransformer("data/rubert_base/")

data_dir = pathlib.Path(__file__).parent / "data"
_, _, answers = load_dataset()
answer_emb = bert_model.encode(answers)
model = tf.keras.models.load_model(data_dir / "model.keras")

print("Добро пожаловать, я могу ответить на ваши вопросы вопросы из жизни университета/факультета/кафедры. Напишите свой вопрос:")
while True:
    question = input("[In]: ").strip().lower()

    if not question:
        print("\n[Error]: Необходимо ввести вопрос для получения ответа:")
    elif question == "exit":
        print("\n[Out]: Программа закончена")
        break
    else:
        q_emb = bert_model.encode([question])[0]

        p = []
        for a_emb in answer_emb:
            qa_emb = np.concatenate([q_emb, a_emb])
            p += [model.predict(np.expand_dims(qa_emb, axis=0), verbose=None)[0, 0]]

        p = np.array(p)
        answer_index = np.argmax(p)
        print("[Out]: ", answers[answer_index], end="\n\n")
