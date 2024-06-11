import numpy as np
import datetime

from .tomi_dataset import TomiDataset


class SapDataset(TomiDataset):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.tipology = None

    def get_dataset(self) -> tuple[list[str], list[str], list[str]]:
        import pandas as pd
        a = pd.read_csv("data/tomi/sap.csv")
        prompts = a["story"].tolist()
        questions = a["question"].tolist()
        answers = a["answer"].tolist()
        self.tipology = a["tipology"].tolist()[:self.n_samples]

        prompts, questions, answers, self.tipology = self.randomize(prompts, questions, answers, self.tipology)
        self.n_samples = min(self.n_samples, len(prompts))

        return prompts, questions, answers

    def save_results(self, model_answers: list[str], correct_answers: list[str]) -> None:
        import pandas as pd
        dataframe = {
            "is_correct": self.evaluate(model_answers, correct_answers),
            "tipology": self.tipology
        }
        df = pd.DataFrame(dataframe)

        accuracy = np.mean(df["is_correct"])
        with open(self.results_file, "a+") as f:
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Test on {self.n_samples} samples {('('+str(self.num_splitted_context)+' splits)' if self.query_method=='cot-wm' else '')}:\n")
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"tipology fact accuracy: {np.mean(df[df['tipology'] == 1]['is_correct'])}\n")
            f.write(f"tipology mind accuracy: {np.mean(df[df['tipology'] == 0]['is_correct'])}\n")

        print(f"accuracy for Sap: {accuracy}")

        if self.has_wandb:
            self.wandb.log({"accuracy": accuracy})
            self.save_wandb_table()