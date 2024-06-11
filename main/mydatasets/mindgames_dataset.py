from .abstract_dataset import AbstractDataset
import pandas as pd
import datetime
import os
from datasets import Dataset

class MindGamesDataset(AbstractDataset, Dataset):
    DATA_SPLIT_PATH = os.path.join("data", "mindgames", "splits.json")
    def __init__(self, args, split_type: str = "test", is_train: bool = False, **kwargs):
            assert "test" in split_type or "train" in split_type or "val", "The filename should contain 'test' or 'train' or 'val'."
            
            super().__init__(args, **kwargs)
            self.split_type = split_type
            self.data_list = []
            self.is_train = is_train
            self.prepare_data()

    def prepare_data(self):
        """ This method is used to prepare the data for the dataset
        """
        problems, questions, answers = self.get_dataset(self.is_train)

        prefix =  """Consider the following description of a situation where some agents interact. 
At the end, I will ask you whether a statement is in entailment or not with the description I gave you.
Here's the description of the situation:
@problem@

This is the end of the description. Now, consider this statement.

Statement: @question@
The statement is in @answer@ with the the description you gave me.
"""

        for i in range(len(problems)):
            # self.data_list.append((problems[i], questions[i], answers[i]))
            self.data_list.append(prefix.replace("@problem@", problems[i]).replace("@question@", questions[i]).replace("@answer@", answers[i]))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def get_dataset(self, is_train: bool = True) -> tuple[list[str], list[str], list[str]]:
        if self.num_splitted_context < 0:
            return self.get_split_dataset() # quick hack to get the split loaded.

        if is_train:
            df = pd.read_csv("./data/mindgames/train-00000-of-00001-29e951c428782278.csv")
        else:
            df = pd.read_csv("./data/mindgames/test-00000-of-00001-7dfe9e22268ffc8b.csv")

        problems = df["premise"].values.tolist()
        questions = df["hypothesis"].values.tolist()
        answers = df["label"].values.tolist()
        problems, questions, answers = self.randomize(problems, questions, answers)
        self.n_samples = min(self.n_samples, len(problems))

        return problems[:self.n_samples], questions[:self.n_samples], answers[:self.n_samples]

    def split_context(self, context: str, max_k: int) -> list[str]:
        return super().split_context(context, max_k, divider=". ")

    def evaluate(self, model_answers: list[str], correct_answers: list[str]) -> list[int]:
        assert len(model_answers) == len(correct_answers), "The number of model answers and correct answers should be the same"
        is_correct = [0] * len(model_answers)
        for i in range(len(model_answers)):
            if model_answers[i] == correct_answers[i]:
                is_correct[i] = 1

        return is_correct
    
    def save_results(self, model_answers: list[str], correct_answers: list[str]) -> None:
        accuracy = sum(self.evaluate(model_answers, correct_answers)) / len(model_answers)
        with open(self.results_file, "a+") as f:
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}\n")
            f.write(f"Split-value: {self.num_splitted_context}")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Test on {self.n_samples} samples{('('+str(self.num_splitted_context)+' splits)' if self.query_method=='cot-wm' else '')}:\n")
            f.write(f"Accuracy: {accuracy}\n")
            if self.query_method == 'tot':
                f.write(f"Generation type: {self.method_generate}\n")

        print(f"accuracy for MindGames: {accuracy}")

        if self.has_wandb:
            self.wandb.log({"accuracy": accuracy})
            self.save_wandb_table()