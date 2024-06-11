import json
import pandas as pd
import numpy as np
import datetime

from .abstract_dataset import AbstractDataset

class OpenTomDataset(AbstractDataset):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.input_type = args.input_type
        assert self.input_type == "attitude", "currently only attidude format is tested"

    def get_dataset(self) -> tuple[list[str], list[str], list[str]]:
        with open('./data/opentom/opentom.json') as f:
            dataset = json.loads(f.read())
        prompts = []
        questions = []
        answers = []
        for sample in dataset:
            if sample['question']['type'] != self.input_type:
                continue
            prompts.append(sample["narrative"])
            questions.append(f"{sample['question']['question']}\n- positive\n- neutral\n- negative")
            answers.append(sample['question']["answer"])

        prompts, questions, answers = self.randomize(prompts, questions, answers)

        self.n_samples = min(self.n_samples, len(prompts))

        return prompts, questions, answers

    def compare_answers(self, model_answer: str, correct_answer: str) -> bool:
        lowered_model_answer = model_answer.lower()
        lowered_correct_answer = correct_answer.lower()
        return lowered_model_answer == lowered_correct_answer or lowered_correct_answer in lowered_model_answer

    def evaluate(self, model_answers: list[str], correct_answers: list[str]) -> list[int]:
        assert len(model_answers) == len(correct_answers), "The number of model answers and correct answers should be the same"
        is_correct = [0] * len(model_answers)
        for i in range(len(model_answers)):
            if self.compare_answers(model_answers[i], correct_answers[i]):
                is_correct[i] = 1

        return is_correct
    
    def save_results(self, model_answers: list[str], correct_answers: list[str]) -> None:
        accuracy = sum(self.evaluate(model_answers, correct_answers)) / len(model_answers)
        with open(self.results_file, "a+") as f:
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}\n")
            f.write(f"Split-value: {self.num_splitted_context}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Test on {self.n_samples} samples {('('+str(self.num_splitted_context)+' splits)' if self.query_method=='cot-wm' else '')}:\n")
            f.write(f"Accuracy: {accuracy}\n")
            if self.query_method == 'tot':
                f.write(f"Generation type: {self.method_generate}\n")

        print(f"accuracy for opentom: {accuracy}")

        if self.has_wandb:
            self.wandb.log({"accuracy": accuracy})
            self.save_wandb_table()