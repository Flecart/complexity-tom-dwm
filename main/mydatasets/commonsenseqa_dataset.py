import os
import datetime
import json
from datasets import load_dataset

from .abstract_dataset import AbstractDataset

class CommonsenseQADataset(AbstractDataset):
    def __init__(self, args, split_type: str = "test", **kwargs):
            assert "train" in split_type or "test" in split_type or "validation" in split_type, "The filename should contain or 'train' or 'dev'."
            
            self.split_type = split_type
            super().__init__(args, **kwargs)

    def get_dataset(self) -> tuple[list[str], list[str], list[str]]:
        prompts, questions, answers = self._generate_examples()

        prompts, questions, answers = self.randomize(prompts, questions, answers)
        self.n_samples = min(self.n_samples, len(prompts))

        return prompts, questions, answers

    def _generate_examples(self):
        dataset = load_dataset("tau/commonsense_qa", split=self.split_type)
        prompts = [""] * len(dataset)
        questions = [""] * len(dataset)
        answers = [""] * len(dataset)

        for i, sample in enumerate(dataset):
            answer = sample["answerKey"]
            choices = sample["choices"]
            question = ""
            for j in range(len(choices["label"])):
                question += f"{choices['label'][j]}: {choices['text'][j]}\n"

            prompt = sample["question"]

            if answer == "":
                print(sample)
                raise ValueError("Answer is empty")
            answers[i] = answer
            prompts[i] = prompt
            questions[i] = question

        print(len(answers), len(prompts), len(questions))
        return prompts, questions, answers

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
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Test on {self.n_samples} samples {('('+str(self.num_splitted_context)+' splits)' if self.query_method=='cot-wm' else '')}:\n")
            f.write(f"Accuracy: {accuracy}\n")

        print(f"accuracy for commonsenseqa: {accuracy}")

        if self.has_wandb:
            self.wandb.log({f"accuracy-commonsenseqa-{self.model_name}-{self.query_method}": accuracy})
            self.save_wandb_table()
