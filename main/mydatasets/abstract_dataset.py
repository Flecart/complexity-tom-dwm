from typing import Callable
import tiktoken
import datetime
from time import sleep
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
import random
import wandb
import json
import main.prompt_utils as prompt_utils
import main.tot_tasks as tot_tasks
from main.utils import _query_model_handle_errors
import main.tot_prompt as tot

QUESTION_PLACEHOLDER = "@question@"
PROBLEM_PLACEHOLDER = "@problem@"
STRUCTURE_PLACEHOLDER = "@structure@"
local_model = None
local_tokenizer = None

class AbstractDataset(metaclass=ABCMeta):
    DATA_SPLIT_PATH=None
    def __init__(self,
                 args,
                 encoder: tiktoken.Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
                **kwargs
        ):
        self.dataset_name = args.dataset_name
        self.config_file = args.config_file
        self.model_name = args.model_name
        self.kshots = args.kshots
        self.n_samples = args.n_samples
        self.random_sample = args.random_sample
        self.query_method = args.query_method
        self.query_position = args.query_position
        self.num_splitted_context = args.num_splitted_context # only for self prompt
        self.seed = args.seed
        self.batch = args.batch
        self.has_wandb = args.has_wandb
        self.method_generate = args.method_generate # only for ToT
        self.args = args

        if self.seed >= 0:
            random.seed(self.seed)

        if self.num_splitted_context < 0:
            assert self.query_method in ["cot-wm-man"], "The query method should be cot-wm-man"
        assert self.query_position in ["beginning", "end"], "The query position should be either 'beginning' or 'end'"

        query_position_suffix = ("" if self.query_position!="beginning" else "-question-first")
        print(f"./prompt/{self.dataset_name}/{self.query_method}-{self.kshots}shot{query_position_suffix}.txt")
        with open(f"./prompt/{self.dataset_name}/{self.query_method}-{self.kshots}shot{query_position_suffix}.txt", "r") as f:
            self.prompt = f.read()

        self.encoder = encoder

        self.log_file = f"./logs/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{self.model_name}-{self.dataset_name}-{self.query_method}-{self.kshots}shot-{self.query_position}.txt"
        self.results_file = f"./results/{self.dataset_name}/{self.query_method}-{self.kshots}shot-{self.query_position}.txt"

        if self.has_wandb:
            self.wandb = wandb.init(project="llm-tom-test", config={
                "dataset_name": args.dataset_name,
                "model_name": args.model_name,
                "kshots": args.kshots,
                "n_samples": args.n_samples,
                "random_sample": args.random_sample,
                "query_method": args.query_method,
                "query_position": args.query_position,
                "num_splitted_context": args.num_splitted_context,
                "seed": args.seed,
                "batch": args.batch,
                # add kwargs and concatenate with this dict
            } | kwargs)
            self.wandb_table = None

    @abstractmethod
    def get_dataset(self, *args, **kwargs) -> tuple[list[str], list[str], list[str]]:
        """ Should return a tuple with the following elements:
        1. A list of input prompts
        2. A list of questions for each input
        3. A list of answers for each input
        """
        pass

    @abstractmethod
    def evaluate(self, model_answers: list[str], correct_answers: list[str], *args, **kwargs):
        """ Evaluates the model answers

        Returns:
        Dataset specific metrics that value the responses of the model
        """
        pass

    @abstractmethod
    def save_results(self, model_answers: list[str], correct_answers: list[str], *args, **kwargs):
        """ Saves the results of the model

        Returns:
        None
        """
        pass

    def get_split_dataset(self) -> tuple[list[str], (list[str], list[tuple[int, int]]), list[str]]:
        if self.DATA_SPLIT_PATH is None:
            raise ValueError("The split path is not defined")
        with open(self.DATA_SPLIT_PATH, "r") as f:
            data = json.load(f)
        prompts, questions, answers = [], [], []
        splits = []
        for d in data:
            prompts.append(d['prompt'])
            questions.append(d['question'])
            answers.append(d['answer'])
            splits.append(d['num_highlights'])

        # print(prompts[:2], questions[:2], answers[:2], splits[:2])
        return prompts, list(zip(questions, splits)), answers

    def write_logs(self, **kwargs) -> None:
        """ Every key in kwargs is a tag and the value is the content written to the logs file
        """
        logs_file_name = self.log_file
        with open(logs_file_name, "a+") as f:
            f.write(f"<sample>\n")
            for key, value in kwargs.items():
                key = key.replace("_", "-")
                f.write(f"<{key}>{value}</{key}>\n")
            f.write(f"</sample>\n")

            if self.has_wandb:
                # this hack is to visualize complex dicts on a single key.
                has_other_info = "other_info" in kwargs
                other_info = None
                if has_other_info and kwargs["other_info"] is not None:
                    other_info = kwargs.pop("other_info")

                if self.wandb_table is None:
                    self.wandb_table = wandb.Table(columns=list(kwargs.keys()) + ["other_info"])

                self.wandb_table.add_data(*list(kwargs.values()), str(other_info))

    def split_context(self, context: str, max_k: int, divider: str = '\n') -> list[str]:
        """ Splits the context into smaller pieces, each with at most max_k phrases
        """
        splitted_context = context.strip().split(divider)
        num_buckets = max_k
        content_per_bucket = (len(splitted_context) + max_k - 1) // max_k
        condensed_splitted_context = [""] * num_buckets
        for j in range(num_buckets):
            start = j * content_per_bucket
            end = min((j + 1) * content_per_bucket, len(splitted_context))
            condensed_splitted_context[j] = divider.join(splitted_context[start:end]).strip()

        return condensed_splitted_context
    
    def self_prompt(self, templates: list[str], splitted_contexts: list[list[str]]):
        """ This is the default self-prompting method for a general dataset
        A children class can override this method if it needs a different self-prompting method

        Args:
        -----
        template: str
            The template to use, it has ONLY a SINGLE problem placeholder.
        splitted_context: list[str]
            The problem splitted into smaller pieces for this prompting method.
        """
        n_batch = len(templates)
        known_texts = [""] * n_batch
        old_answers = [""] * n_batch
        prompts = [""] * n_batch

        max_split_len = max([len(context) for context in splitted_contexts])

        for i in range(max_split_len):
            # This logic may seem a little bit complex, but I decided to handle contexts 
            # with different lengths here, so that i batched only the bases where the context is still valid
            # used k to handle the different indexes counters

            # check how many prompts are still valid
            valid_prompts = [True if i < len(splitted_contexts[k]) else False for k in range(n_batch)]
            prompts = [""] * sum(valid_prompts)

            # prepare batch
            k = 0  # valid counter
            for j in range(n_batch):
                if not valid_prompts[j]:
                    continue

                known_texts[j] += f"{splitted_contexts[j][i]}\n"
                prompts[k] = templates[j].replace(PROBLEM_PLACEHOLDER, known_texts[j])
                k += 1

            outputs = _query_model_handle_errors(prompts, self.config_file, self.model_name, self.encoder)

            assert len(outputs) == sum(valid_prompts), "The number of outputs should be the same as the number of prompts"
            k = 0
            for j in range(n_batch):
                if not valid_prompts[j]:
                    continue

                try:
                    new_knowledge = prompt_utils.delete_tag(outputs[k], "answer").strip()
                    known_texts[j] += prompt_utils.add_comment(new_knowledge) + "\n"
                    answer = prompt_utils.extract_tag(outputs[k], "answer")
                except:
                    answer = ""

                if len(answer) > 0:
                    old_answers[j] = answer.lower().strip()
                else:
                    old_answers[j] = "<No answer>"  # so we save the whole response, and not just the empty string...

                k += 1
        
        return old_answers, known_texts
    
    def self_prompt_chat(self, templates: list[str], splitted_contexts: list[list[str]]):
        """ This is the default self-prompting method for a chat prompting
        A children class can override this method if it needs a different self-prompting method

        Args:
        -----
        template: str
            The template to use, it has ONLY a SINGLE problem placeholder.
        splitted_context: list[str]
            The problem splitted into smaller pieces for this prompting method.
        """
        n_batch = len(templates)
        known_texts = [[] for _ in range(n_batch)]
        old_answers = [""] * n_batch

        max_split_len = max([len(context) for context in splitted_contexts])
        for i in range(max_split_len):
            # This logic may seem a little bit complex, but I decided to handle contexts 
            # with different lengths here, so that i batched only the bases where the context is still valid
            # used k to handle the different indexes counters

            # check how many prompts are still valid
            valid_prompts = [True if i < len(splitted_contexts[k]) else False for k in range(n_batch)]
            prompts = [""] * sum(valid_prompts)

            # prepare batch
            k = 0  # valid counter
            for j in range(n_batch):
                if not valid_prompts[j]:
                    continue
                if i == 0:
                    known_texts[j].append({
                        "role": "user",
                        "content": templates[j].replace(PROBLEM_PLACEHOLDER, splitted_contexts[j][i])
                    })
                else:
                    known_texts[j].append({
                        "role": "user",
                        "content": splitted_contexts[j][i]
                    })

                prompts[k] = known_texts[j]
                k += 1

            outputs = _query_model_handle_errors(prompts, self.config_file, self.model_name, self.encoder)

            assert len(outputs) == sum(valid_prompts), "The number of outputs should be the same as the number of prompts"
            k = 0
            for j in range(n_batch):
                if not valid_prompts[j]:
                    continue

                try:
                    # print(f"current output -{outputs[k]}- -{prompt_utils.extract_tag(outputs[k], 'answer')}-")
                    new_knowledge = prompt_utils.delete_tag(outputs[k], "answer").strip()
                    known_texts[j].append({
                        "role": "assistant",
                        "content": new_knowledge
                    })
                    answer = prompt_utils.extract_tag(outputs[k], "answer")
                except:
                    answer = ""

                if len(answer) > 0:
                    old_answers[j] = answer.lower().strip()
                else:
                    old_answers[j] = "<No answer>"  # so we save the whole response, and not just the empty string...
                
                k += 1

        return old_answers, known_texts

    def run_experiments(self, problems: list[str], questions: list[str], answers: list[str] = None) -> list[str]:
        """ Runs single examples with the given sample_runner

        Runner is a function that evaluates a single problem sample.

        Note: if you need a different runner, you should override 
        the corresponding runner method on the child class.
        """

        sample_runner:  Callable[[str, str], str] = None
        match self.query_method:
            case "cot-wm":
                sample_runner = self._cot_wm_runner
            case "cot-wm-chat":
                sample_runner = self._cot_wm_chat_runner
            case "cot-wm-chat-repeat":
                sample_runner = self._cot_wm_chat_repeat_runner
            case "cot-wm-man":
                sample_runner = self._cot_wm_man_runner
            case "tot":
                tot.init_gpt(self.config_file, self.model_name)
                sample_runner = self._tot_runner
            case "cot":
                sample_runner = self._cot_runner
            case "cot-spaced":
                sample_runner = self._cot_spaced_runner
            case "struct":
                sample_runner = self._struct_runner
            case "cot-explwm":
                sample_runner = self._cot_wm_runner
            case "struct-yaml":
                sample_runner = self._struct_runner
            case _:
                raise ValueError("Invalid query method")

        assert len(problems) == len(questions), "The number of problems and questions should be the same"
        if self.query_method == "cot-wm-man": self.n_samples = 50
        else: assert len(problems) == self.n_samples, "The number of problems should be the same as the number of samples we are considering"

        model_answers = [""] * len(problems)
        pbar = tqdm(total=(self.n_samples + self.batch - 1) // self.batch)
        for i in range(0, len(problems), self.batch):
            curr_problems = problems[i:i+self.batch]
            curr_questions = questions[i:i+self.batch]

            curr_answers, other_info = sample_runner(curr_problems, curr_questions)
            pbar.update(1)
            for j in range(len(curr_answers)):
                model_answers[i+j] = curr_answers[j]

                if self.query_method == "cot-wm-man": # small hack to make the following work:
                    questions[i+j] = questions[i+j][0]

                self.write_logs(
                    index=i+j,
                    problem=problems[i+j],
                    other_info=other_info[j] if other_info is not None else None,
                    question=questions[i+j],
                    model_answer=curr_answers[j],
                    correct_answer=answers[i+j] if answers is not None else None,
                    is_same=str(self.compare_answers(curr_answers[j], answers[i+j])) if answers is not None else None,
                )
        pbar.close()

        return model_answers
    
    def compare_answers(self, model_answer: str, correct_answer: str) -> bool:
        """ Default function used to compare the model answer with the correct answer

        Needs to be overriden if the dataset has a different way of comparing answers
        """
        return model_answer == correct_answer

    def randomize(self, *args) -> list:
        """ gets many lists of the same length in input, returns a permutation of the indices
        Assumes that all the lists have the same length
        """
        if len(args) == 0:
            return []

        length = len(args[0])
        for arg in args:
            assert len(arg) == length, "All the lists should have the same length"

        max_samples = min(self.n_samples, length)
        indices = (random.sample(range(length), max_samples) if self.random_sample else [i for i in range(max_samples)])
        result = []
        for arg in args:
            result.append([arg[i] for i in indices])

        return result

    def _cot_wm_runner(self, problems: list[str], questions: list[str]) -> tuple[str, str]:
        prompt_without_questions = [""] * len(problems)
        splitted_contexts = [""] * len(problems)

        for i in range(len(problems)):
            problem = problems[i]
            question = questions[i]

            splitted_context = self.split_context(problem, self.num_splitted_context)
            model_prompt = self.prompt.replace(QUESTION_PLACEHOLDER, question)
            prompt_without_question, question_suffix = model_prompt.split('====')
            splitted_context.append(question_suffix)

            prompt_without_questions[i] = prompt_without_question
            splitted_contexts[i] = splitted_context

        answers, known_texts = self.self_prompt(prompt_without_questions, splitted_contexts)

        return answers, known_texts

    def _cot_wm_man_runner(self, problems: list[str], questions_splits: tuple[list[str], list[tuple[int, int]]]) -> tuple[str, str]:
        prompt_without_questions = [""] * len(problems)
        splitted_contexts = [""] * len(problems)

        for i in range(len(problems)):
            problem = problems[i]
            question, splits = questions_splits[i]

            model_prompt = self.prompt.replace(QUESTION_PLACEHOLDER, question)
            prompt_without_question, question_suffix = model_prompt.split('====')

            prompt_without_questions[i] = prompt_without_question
            splitted_contexts[i] = []

            end = 0
            current_split_content = ""
            for split in splits:
                current_split_content += problem[end:split[0]]
                current_split_content += "Describe this part with more attention: " + problem[split[0]:split[1]]
                splitted_contexts[i].append(current_split_content)
                end = split[1]
                current_split_content = ""
            current_split_content += problem[end:]
            current_split_content += question_suffix
            splitted_contexts[i].append(current_split_content)

        answers, known_texts = self.self_prompt_chat(prompt_without_questions, splitted_contexts)

        return answers, known_texts

    def _tot_runner(self, problems: list[str], questions: list[str]) -> tuple[str, str]:

        answers = [""] * len(problems)
        known_texts = [""] * len(problems)

        match self.dataset_name:
            case "tomi":
                task = tot_tasks.TomiTask(database_name=self.dataset_name, kshot=self.kshots)
            case "adv-csfb":
                task = tot_tasks.AdvCsfbTask(database_name=self.dataset_name, kshot=self.kshots)
            case "fantom":
                task = tot_tasks.FantombTask(database_name=self.dataset_name, kshot=self.kshots)
            case "mindgames":
                task = tot_tasks.MindgamesTask(database_name=self.dataset_name, kshot=self.kshots)
            case "socialiqa":
                task = tot_tasks.SocialIQaTask(database_name=self.dataset_name, kshot=self.kshots)
            case "commonsenseqa":
                task = tot_tasks.CommonsenseQATask(database_name=self.dataset_name, kshot=self.kshots)
            case _:
                raise ValueError("Invalid or unsupported dataset name")

        for i in range(len(problems)):
            # print(problems[i])
            # print(questions[i])
            problem = problems[i] + "\n" + questions[i]
            ys, info = tot.solve(self.args, task, problem)

            answer = task.test_output(problems[i], questions[i], ys)

            try:
                answers[i] = prompt_utils.extract_tag(answer, "answer")
            except:
                answers[i] = ""
            known_texts[i] = info

        return answers, known_texts

    def _tot_old_runner(self, problems: list[str], questions: list[str]) -> tuple[str, str]:
        """ This runner is deprecated it used the approach of ToT explained in 
        https://github.com/dave1010/tree-of-thought-prompting
        But it is too different
        """
        answers = [""] * len(problems)
        known_texts = [""] * len(problems)

        def translate_int(i: int) -> str:
            if i == 2:
                return "two"
            elif i == 3:
                return "three"
            elif i == 4:
                return "four"
            elif i == 5:
                return "five"
            else:
                raise ValueError("Invalid number of splits")

        for i in range(len(problems)):
            problem = problems[i]
            question = questions[i]

            model_prompt = self.prompt.replace(QUESTION_PLACEHOLDER, question)
            model_prompt = model_prompt.replace(PROBLEM_PLACEHOLDER, problem)
            model_prompt = model_prompt.replace("@experts@", translate_int(self.num_splitted_context))

            answer = _query_model_handle_errors([model_prompt], self.config_file, self.model_name, self.encoder)[0]

            try:
                answers[i] = prompt_utils.extract_tag(answer, "answer")
            except:
                answers[i] = ""
            known_texts[i] = model_prompt + answer
            print(known_texts[i])

            # print(answer, answers[i], model_prompt)

        return answers, known_texts
    

    def _cot_wm_chat_runner(self, problems: list[str], questions: list[str]) -> list[tuple[str, str]]:
        prompt_without_questions = [""] * len(problems)
        splitted_contexts = [""] * len(problems)

        for i in range(len(problems)):
            problem = problems[i]
            question = questions[i]

            splitted_context = self.split_context(problem, self.num_splitted_context)
            model_prompt = self.prompt.replace(QUESTION_PLACEHOLDER, question)
            prompt_without_question, question_suffix = model_prompt.split('====')
            splitted_context.append(question_suffix)

            prompt_without_questions[i] = prompt_without_question
            splitted_contexts[i] = splitted_context

        answers, known_texts = self.self_prompt_chat(prompt_without_questions, splitted_contexts)

        return answers, known_texts
    
    def _cot_wm_chat_repeat_runner(self, problems: list[str], questions: list[str]) -> list[tuple[str, str]]:
        prompt_without_questions = [""] * len(problems)
        splitted_contexts = [""] * len(problems)

        for i in range(len(problems)):
            problem = problems[i]
            question = questions[i]

            splitted_context = self.split_context(problem, self.num_splitted_context)
            model_prompt = self.prompt.replace(QUESTION_PLACEHOLDER, question)
            model_prompt = model_prompt.replace("@text@", problem)
            prompt_without_question, question_suffix = model_prompt.split('====')
            splitted_context.append(question_suffix)

            prompt_without_questions[i] = prompt_without_question
            splitted_contexts[i] = splitted_context

        answers, known_texts = self.self_prompt_chat(prompt_without_questions, splitted_contexts)

        return answers, known_texts


    def _cot_spaced_runner(self, problems: list[str], questions: list[str]) -> tuple[str, None]:
        """ Ablation study to break the structure of the problem into smaller pieces"""
        new_problems = [""] * len(problems)

        for i, problem in enumerate(problems):
            splitted_context = self.split_context(problem, self.num_splitted_context)

            ignore_sentence = "\n" + "AAAAAAAAAAAAAAAAAAAAAA\n"*1
            new_problem = ignore_sentence.join(splitted_context)
            new_problems[i] = new_problem

        return self._cot_runner(new_problems, questions)

    def _cot_runner(self, problems: list[str], questions: list[str]) -> tuple[str, None]:
        model_prompts = ["" for _ in range(len(problems))]
        answers = [""] * len(problems)

        for i in range(len(problems)):
            model_prompt = self.prompt.replace(QUESTION_PLACEHOLDER, questions[i])
            model_prompt = model_prompt.replace(PROBLEM_PLACEHOLDER, problems[i])
            model_prompts[i] = model_prompt

        output = _query_model_handle_errors(model_prompts, self.config_file, self.model_name, self.encoder)

        for i in range(len(problems)):
            try:
                answers[i] = prompt_utils.extract_tag(output[i], "answer")
            except:
                answers[i] = ""

        return answers, model_prompts
    
    def _struct_runner(self, problems: list[str], questions: list[str]) -> tuple[str, str]:
        first_prompts = ["" for _ in range(len(problems))]
        second_prompts = ["" for _ in range(len(problems))]
        answers = [""] * len(problems)

        for i in range(len(problems)):
            current_prompt = self.prompt.replace(QUESTION_PLACEHOLDER, questions[i])
            current_prompt = current_prompt.replace(PROBLEM_PLACEHOLDER, problems[i])
            first_prompt, second_prompt = current_prompt.split("====")
            first_prompts[i] = first_prompt
            second_prompts[i] = second_prompt

        first_outputs = _query_model_handle_errors(first_prompts, self.config_file, self.model_name, self.encoder)
        
        for i in range(len(problems)):
            first_output = first_outputs[i]
            second_prompts[i] = second_prompts[i].replace(STRUCTURE_PLACEHOLDER, first_output)

        second_output = _query_model_handle_errors(second_prompts, self.config_file, self.model_name, self.encoder)
        
        for i in range(len(problems)):
            try:
                answers[i] = prompt_utils.extract_tag(second_output[i], "answer")
            except:
                answers[i] = ""

        return answers, first_outputs

    def save_wandb_table(self):
        if self.has_wandb and self.wandb_table is not None:
            artifact = wandb.Artifact(f"{self.dataset_name}-{self.query_method}-{self.kshots}shot-{self.query_position}", type="dataset")
            artifact.add(self.wandb_table, "results")
            self.wandb.log_artifact(artifact)