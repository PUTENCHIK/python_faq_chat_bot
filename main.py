import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
tf.get_logger().setLevel('NOTSET')

from funcs import load_dataset


if __name__ == "__main__":
    _, _, answers = load_dataset()
    
    bert_prepr = "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"
    bert_model = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3"

    print("BERT Preprocessor loading..")
    bert_prepr = hub.KerasLayer(bert_prepr)
    print("BERT Model loading..")
    bert_model = hub.KerasLayer(bert_model)

    data_dir = pathlib.Path(__file__).parent / "data"
    print("Model loading..")
    model = tf.keras.models.load_model(data_dir / "model.keras")
    print("Data loading..")
    data = np.load(data_dir / "data.npy")

    print("Вводите вопросы об университете. При вводе [exit] программа закончится.")
    while True:
        question = input("[In]: ")

        if not question:
            print("\n[Error]: Необходимо ввести вопрос для получения ответа:")
        elif question == "exit":
            print("\n[Out]: Программа закончена")
            break
        else:
            q_emb = bert_model(bert_prepr([question.lower()]))["pooled_output"].numpy()[0]
            p = []
            for i in range(data.shape[0]):
                a_emb = data[i, 1]
                qa_emb = np.concatenate([q_emb, a_emb])
                p += [model.predict(np.expand_dims(qa_emb, axis=0), verbose=None)[0, 0]]

            p = np.array(p)
            answer_index = np.argmax(p)
            print("[Out]: ", answers[answer_index], end="\n\n")
