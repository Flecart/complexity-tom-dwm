import json
import pandas as pd
import numpy as np
import datetime
import os

from .abstract_dataset import AbstractDataset

class AdvCsfbDataset(AbstractDataset):
    DATA_SPLIT_PATH = os.path.join("data", "Adv-CSFB", "splits.json")

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.output_type = args.output_type

    def get_split_dataset(self):
        problems, questions, anwers = super().get_split_dataset()
        return problems, questions, None, anwers, None

    def get_dataset(self) -> tuple[list[str], list[str], list[str]]:
        if self.num_splitted_context < 0:
            return self.get_split_dataset() # quick hack to get the split loaded.

        data = []
        with open('./data/Adv-CSFB/unexpected_contents.jsonl') as f:
            for line in f:
                data.append(json.loads(line))
        with open('./data/Adv-CSFB/unexpected_contents_source.json') as f:
            names = json.load(f)['data']

        problems, questions, options, answers = [], [], [], []
        type_value = []
        for i, d in enumerate(data):
            problems.append(d['txt'])
            for tag in d.keys():
                problems[-1] = problems[-1].replace(f'[{tag}]', str(d[tag]))
            problems[-1] += names[i % len(names)]['xpro'].capitalize() + ' is delighted to have found this ' + d['ctr'] + '.'
            options.append((d['o1'].lower().strip(), d['o2'].lower().strip()))  
            if self.output_type == 'open':
                questions.append("\nNow, complete the dialogue with a short sentence that logically follows up what is described in the dialogue.")
            elif self.output_type == 'multiple':
                questions.append('Question: Fill in the blank with the best option. ' + names[i % len(names)]['xpro'].capitalize() + ' ' + d['q3'].strip() + " _" + f"\n- {options[-1][0]}\n- {options[-1][1]}") 
            else:
                raise Exception(f"{self.output_type} is not a valid self.output_type value.")        
            answers.append(d[d['truth']])  
            type_value.append(d['type'])

        problems, questions, options, answers, type_value = self.randomize(problems, questions, options, answers, type_value)
        self.n_samples = min(self.n_samples, len(problems))

        # print(f"Dataset: Adv-csfb choice questions")
        # print(f"Number of samples: {self.n_samples}")
        # print(f"Output type: {self.output_type}")
        # print(problems[:1])
        # print(questions[:1])
        # print(options[:1])
        # print(answers[:1])
        # print(type_value[:1])
        return problems, questions, options, answers, type_value

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
    
    def save_results(self, model_answers: list[str], correct_answers: list[str], types: list[str] = None) -> None:
        evaluation = self.evaluate(model_answers, correct_answers)
  

        accuracy = np.mean(evaluation)
        print(f"accuracy for Adv-csfb choice questions: {accuracy}")
        with open(self.results_file, "a+") as f:
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}\n")
            f.write(f"Output type: {self.output_type}")
            f.write(f"Split-value: {self.num_splitted_context}")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Test on {self.n_samples} samples {('('+str(self.num_splitted_context)+' splits)' if self.query_method=='cot-wm' else '')}:\n")
            if self.query_method == 'tot':
                f.write(f"Generation type: {self.method_generate}\n")
            f.write(f"All Accuracy: {accuracy}\n")


            if types is not None:
                dataframe = {
                    "evaluation": evaluation,
                    "types": types
                }

                df = pd.DataFrame(dataframe)

                # create results based on the type of question
                different_types = df['types'].unique()
                type_map = {
                    "fb": "False Belief",
                    "late label": "Late Label",
                    "tb": "True Belief",
                    "transparent access": "Transparent Access",
                    "trusted testimony": "Trusted Testimony",
                    "uninformative label": "Uninformative Label",
                }
                for t in different_types:
                    f.write(f"{type_map[t]} accuracy: {np.mean(df[df['types'] == t]['evaluation'])}\n")
                f.write("\n")

        if self.has_wandb:
            self.wandb.log({"accuracy": accuracy})
            self.save_wandb_table()