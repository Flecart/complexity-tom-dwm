""" This file copies the statefulness data into the main data directory 
and keeps only the correct ones.
"""
from nltk import tokenize
import os
import json
from pydantic import BaseModel
import numpy as np
import nltk

nltk.download('punkt')

data_dir = "statefulness/data/"
data_map = {
    "adv-csfb": "Adv-CSFB",
    "fantom": "fantom",
    "mindgames": "mindgames",
    "socialiqa": "socialIQa",
    "tomi": "tomi",
}

state_multiplier = {
    "adv-csfb": 1,
    "fantom": 1,
    "mindgames": 1,
    "socialiqa": 1,
    "tomi": 1,
}

output_dir = "data/"

# cicle all files inside data_dir
print(data_dir)


class Sample(BaseModel):
    prompt: str
    question: str
    answer: str
    num_states: int = -1
    num_highlights: list[tuple[int, int]] = []

def process_json(data: list[Sample]) -> list[Sample]:
    final_list = []
    for d in data:
        if d.num_states != -1 and len(d.num_highlights) == d.num_states:
            final_list.append(d)

    return final_list

def get_number_sentences(data: list[Sample]) -> np.ndarray[int]:
    num_sentences = []
    for d in data:
        num_sentences.append(len(tokenize.sent_tokenize(d.prompt)))

    return np.array(num_sentences)

for files in os.listdir(data_dir):
    print(files)
    if files.endswith(".json"):
        with open(os.path.join(data_dir, files), "r") as f:
            data = json.load(f)
        processsed = process_json([Sample(**d) for d in data])
        # print(processsed[:2], len(processsed))

        end_path = os.path.join(output_dir, data_map[files.split(".")[0]], "splits.json")
        print(end_path)

        with open(end_path, "w") as f:
            json.dump([d.model_dump() for d in processsed], f, indent=4)

        # calculate statefulness value
        import numpy as np
        statefulness = np.array([d.num_states for d in processsed])
        print(f"Statefulness value for {files}: {statefulness.mean()}, std: {statefulness.std()}")
        print(list(statefulness))
        print("\n")

        print(f"File: {files}")
        print(statefulness)
        
        # calculate statelessness value
        num_sentences = get_number_sentences(processsed) - statefulness
        print(f"Statelessness value for {files}: {num_sentences.mean()}, std: {num_sentences.std()}")
        print(list(num_sentences))

        print(f"Total mean cost: {statefulness.mean() + num_sentences.mean()}")
        print()

            # if file.endswith(".jsonl"):
            #     with open(os.path.join(root, file), "r") as f:
            #         data = json.load(f)
            #     new_data = []
            #     for d in data:
            #         if d["correct"]:
            #             new_data.append(d)
            #     with open(os.path.join(root, file), "w") as f:
            #         json.dump(new_data, f, indent=4)


