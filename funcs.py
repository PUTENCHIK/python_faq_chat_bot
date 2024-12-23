import json
import pathlib


def load_dataset() -> tuple:
    data_dir = pathlib.Path(__file__).parent / "data"

    with open(data_dir / "dataset.json", encoding="utf-8") as file:
        dataset = json.load(file)

    questions, answers = [], []
    for item in dataset:
        questions += [item['q'].lower()]
        answers += [item['a'].lower()]
    
    return dataset, questions, answers
