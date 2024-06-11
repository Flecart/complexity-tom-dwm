import os
import datetime
import json

from .abstract_dataset import AbstractDataset

def _delete_first_word(sample: str) -> str:
    """ Deletes the first word in every line of the string
    
    Returns:
    str: The sample without the first word in every line
    """
    lines = sample.split("\n")

    for i in range(len(lines)):
        space_index = lines[i].index(" ")
        lines[i] = lines[i][space_index+1:]
    
    return "\n".join(lines)

def _extract_components(sample: str) -> tuple[str, str, str]:
    """ Extracts the components of the sample
    First lines are the prompt
    Last line is the question and the answer

    Returns:
    tuple[str, str, str]: The prompt, the question and the answer
    """
    lines = sample.split("\n")
    prompt = "\n".join(lines[:-1])
    question_and_answer = lines[-1].strip().split("\t")
    question = question_and_answer[0]
    answer = question_and_answer[1]
    return prompt, question, answer

class TomiDataset(AbstractDataset):
    DATA_SPLIT_PATH = os.path.join("data", "tomi", "splits.json")
    def __init__(self, args, split_type: str = "test", **kwargs):
            assert "test" in split_type or "train" in split_type or "val", "The filename should contain 'test' or 'train' or 'val'."
            
            self.split_type = split_type
            super().__init__(args, **kwargs)

    def get_dataset(self) -> tuple[list[str], list[str], list[str]]:
        if self.num_splitted_context < 0:
            return self.get_split_dataset() # quick hack to get the split loaded.

        split = self.split_type

        filename = os.path.join("data", "tomi", split + ".txt")
        with open(filename, "r") as f:
            s = f.read()
            prompts = s.strip().split("1\n")
            prompts = [_delete_first_word(prompt) for prompt in prompts]

            prompts, questions, answers = zip(*[_extract_components(prompt) for prompt in prompts])
            answers = [a.lower().strip() for a in answers]  # lower-case and no spaces
            prompts, questions, answers = self.randomize(prompts, questions, answers)

            self.n_samples = min(self.n_samples, len(prompts))

            return prompts, questions, answers

    def compare_answers(self, model_answer: str, correct_answer: str) -> bool:
        return correct_answer in model_answer.lower().replace(" ", "_").replace("\\", "")

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

        print(f"accuracy for Tomi: {accuracy}")

        if self.has_wandb:
            self.wandb.log({"accuracy": accuracy})
            self.save_wandb_table()