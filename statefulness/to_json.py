"""Extracts all databases and saves them into json pydantic formats"""
from main.mydatasets import FantomDataset, TomiDataset, AdvCsfbDataset, MindGamesDataset, SocialIQaDataset
from pydantic import BaseModel, RootModel

class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# just a copied version for logs
args = {
    "config_file": "./config-azure.json",
    "model_name": "gpt-3.5-azure",
    "query_method": "cot",
    "query_position": "end",
    "kshots": "0",
    "n_samples": 1000,
    "random_sample": "True",
    "dataset_name": "tomi",
    "num_splitted_context": 1,
    "seed": 42,
    "input_type": "attitude",
    "output_type": "multiple",
    "batch": 1,
    "has_wandb": False,
    "n_generate_sample": 3,
    "method_generate": "sample",
    "method_evaluate": "vote",
    "method_select": "greedy",
    "n_evaluate_sample": 1,
    "n_select_sample": 1,
}

class Sample(BaseModel):
    prompt: str
    question: str
    answer: str
    num_states: int = -1
    num_highlights: list[tuple[int, int]] = []

class SampleList(RootModel):
    root: list[Sample]

args = dotdict(args)

# Save to json
import json

base_dir = "./statefulness/data"

datasets = [TomiDataset, AdvCsfbDataset, MindGamesDataset, SocialIQaDataset, FantomDataset]
names = ["tomi", "adv-csfb", "mindgames", "socialiqa", "fantom"]


for dataset, name in zip(datasets, names):
    if name != "fantom":
        continue

    with open(f"{base_dir}/{name}.json", "w") as f:

        dataset = dataset(args)
        if name != "adv-csfb":
            prompts, questions, answers = dataset.get_dataset()

            if name == 'fantom':
                answers = [f"Good: {answers[i][0]}\nBad: {answers[i][1]}" for i in range(len(answers))]
        else:
            prompts, questions, _, answers, _ = dataset.get_dataset()

        all_samples = []
        for prompt, question, answer in zip(prompts, questions, answers):
            all_samples.append(Sample(prompt=prompt, question=question, answer=answer).model_dump())

        json.dump(all_samples, f, indent=4)