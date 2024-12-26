from sentence_transformers import SentenceTransformer


if __name__ == "__main__":
    print(f"BERT model setuping..")
    bert_model = SentenceTransformer("DeepPavlov/rubert-base-cased")
    print(f"BERT model setup completed")

    print(f"Saving BERT model..")
    bert_model.save("data/rubert_base/")
    print("BERT model saved")
