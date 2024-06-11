import os
import datetime
import json

from .abstract_dataset import AbstractDataset

class SocialIQaDataset(AbstractDataset):
    def __init__(self, args, split_type: str = "dev", **kwargs):
            assert "train" in split_type or "dev" in split_type, "The filename should contain or 'train' or 'dev'."
            
            self.split_type = split_type
            super().__init__(args, **kwargs)

    def get_dataset(self) -> tuple[list[str], list[str], list[str]]:
        if self.num_splitted_context < 0:
            return self.get_split_dataset() # quick hack to get the split loaded.

        split = self.split_type
        filepath = os.path.join("data",  "socialIQa", f"{split}.jsonl")
        labelpath = os.path.join("data", "socialIQa" , f"{split}-labels.lst")

        prompts, questions, answers = self._generate_examples(filepath, labelpath)

        prompts, questions, answers = self.randomize(prompts, questions, answers)
        self.n_samples = min(self.n_samples, len(prompts))

        return prompts, questions, answers

    def _generate_examples(self, filepath, labelpath):
        """Yields examples."""
        # TODO(social_i_qa): Yields (key, example) tuples from the dataset
        with open(labelpath, encoding="utf-8") as f:
            labels = [label.strip() for label in f]

        prompts = [""] * len(labels)
        questions = [""] * len(labels)
        answers = [""] * len(labels)

        with open(filepath, encoding="utf-8") as f1:
            for i, row in enumerate(f1):
                data = json.loads(row)
                label = labels[i]
                context = data["context"]
                answerA = data["answerA"]
                answerB = data["answerB"]
                answerC = data["answerC"]
                question = data["question"]
                
                answers[i] = str(label)
                prompts[i] = f"{context}"
                questions[i] = f"{question}\n1: {answerA}\n2: {answerB}\n3: {answerC}"

        return prompts, questions, answers

    def compare_answers(self, model_answer: str, correct_answer: str) -> bool:
        return correct_answer in model_answer # the answer could be 2: answer

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
            f.write(f"Split-value: {self.num_splitted_context}")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Test on {self.n_samples} samples {('('+str(self.num_splitted_context)+' splits)' if self.query_method=='cot-wm' else '')}:\n")
            f.write(f"Accuracy: {accuracy}\n")
            if self.query_method == 'tot':
                f.write(f"Generation type: {self.method_generate}\n")

        print(f"accuracy for socialIQa: {accuracy}")

        if self.has_wandb:
            self.wandb.log({f"accuracy": accuracy})
            self.save_wandb_table()