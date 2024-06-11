import numpy as np
import datetime
import os

from .abstract_dataset import AbstractDataset
from .fantom_eval_agent import FantomEvalAgent

class FantomDataset(AbstractDataset):
    DATA_SPLIT_PATH = os.path.join("data", "fantom", "splits.json")
    class SampleArgs:
        aggregation_target = "no"
        conversation_input_type = "short"

    def __init__(self, 
                 args,
                 aggregation_target: str = "no", 
                 conversation_input_type: str = "short", 
                 type_keys: list[str] = ["tom:belief:inaccessible", "tom:belief:accessible"],
                 **kwargs):
        super().__init__(args, **kwargs)

        FantomDataset.SampleArgs.aggregation_target = aggregation_target
        FantomDataset.SampleArgs.conversation_input_type = conversation_input_type

        self.fantom_eval_agent = FantomEvalAgent(FantomDataset.SampleArgs())
        self.fantom_eval_agent.load_fantom()
        self.fantom_eval_agent.setup_fantom()
        self.flattened_fantom = self.fantom_eval_agent.flattened_fantom
        self.flattened_fantom = list(filter(lambda x: x["question_type"] in type_keys, self.flattened_fantom))

    def get_split_dataset(self):
        prompts, question_splits, answers = super().get_split_dataset()

        goods = []
        bads = []

        for answer in answers:
            good, bad = answer.split('\n')

            good = good[len("good: "):]
            bad = bad[len("bad: "):]
            goods.append(good)
            bads.append(bad)

        return prompts, question_splits, list(zip(goods, bads))


    def compare_answers(self, model_answer: str, correct_answer: tuple[str, str]) -> tuple[bool, float]:
        """Overrides the default compare_answers method to use the FantomEvalAgent"""
        tmp_dict = {
            "correct_answer": correct_answer[0],
            "wrong_answer": correct_answer[1],
        }

        return self.fantom_eval_agent.evaluate_belief_q(tmp_dict, model_answer)

    def get_dataset(self) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        if self.num_splitted_context < 0:
            return self.get_split_dataset() # quick hack to get the split loaded.

        prompts = [self.flattened_fantom[i]["context"] for i in range(self.n_samples)]
        questions = [self.flattened_fantom[i]["question"] for i in range(self.n_samples)]
        answers = [(self.flattened_fantom[i]["correct_answer"], self.flattened_fantom[i]["wrong_answer"]) for i in range(self.n_samples)]
        self.n_samples = min(self.n_samples, len(prompts))

        prompts, questions, answers = self.randomize(prompts, questions, answers)
        return prompts, questions, answers

    def evaluate(self, model_answers: list[str], correct_answers: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        assert len(model_answers) == len(correct_answers), "The number of model answers and correct answers should be the same"

        is_corrects = np.zeros((self.n_samples), dtype=int)
        f1_scores = np.zeros((self.n_samples))

        for j in range(self.n_samples):
            is_corrects[j], f1_scores[j] = self.compare_answers(model_answers[j], correct_answers[j])

        return is_corrects, f1_scores
    
    def save_results(self, model_answers: list[str], correct_answers: list[tuple[str, str]]) -> None:
        is_corrects, f1_scores = self.evaluate(model_answers, correct_answers)
        accuracy = np.mean(is_corrects)
        f1 = np.mean(f1_scores)

        with open(self.results_file, "a+") as f:
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}\n")
            f.write(f"Split-value: {self.num_splitted_context}")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Test on {self.n_samples} samples {('('+str(self.num_splitted_context)+' splits)' if self.query_method=='cot-wm' else '')}:\n")
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"F1: {f1}\n")
            if self.query_method == 'tot':
                f.write(f"Generation type: {self.method_generate}\n")

        print(f"accuracy for Fantom: {accuracy}")

        if self.has_wandb:
            self.wandb.log({"accuracy": accuracy})
            self.save_wandb_table()