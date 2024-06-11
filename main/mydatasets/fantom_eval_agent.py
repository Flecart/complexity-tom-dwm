# Adapted from https://github.com/skywalker023/fantom/blob/main/eval_fantom.py

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import loader
import random

from collections import Counter

random.seed(42)

class FantomEvalAgent():
    def __init__(self, args):
        self.args = args
        # self.prompt_header = "This is a theory-of-mind test. Please answer the question regarding facts or beliefs, based on the following in-person conversation between individuals who have just met.\n\n"
        # self.output_filename_suffix = '_{}_input_{}_cot-{}.json'.format(self.args.conversation_input_type, self.args.model, self.args.use_cot)
        # self.load_fantom()
        # self.setup_fantom()

        # self.model = self.load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("loading embedder")
        import time
        start = time.time()
        self.embedder = SentenceTransformer('sentence-transformers/all-roberta-large-v1', cache_folder=".cache/huggingface/hub").to(self.device)
        print("loaded embedder")
        print("Time taken to load embedder: ", time.time() - start)

    def load_fantom(self):
        self.fantom_df = loader.load()

    def respond(self, prompt):
        response = self.model.interact(prompt)
        return response

    def evaluate_fact_q(self, qa, model_response):
        result = self.compute_f1(qa['correct_answer'].lower(), model_response.lower())
        return result

    def evaluate_belief_q(self, qa, model_response, metric='cosine'):
        """
        Evaluate the belief question by comparing the model's response with the correct answer and wrong answer.

        Args:
            qa (dict): A dictionary containing the question and answers.
            model_response (str): The model's response to the question.
            metric (str, optional): The similarity metric to use for comparison. Defaults to 'cosine'.

        Returns:
            tuple: A tuple containing a boolean value indicating if the model's response matches the correct answer,
                   and the lexical overlap score between the model's response and the corresponding answer.
        """
        wrong_tom_view = qa['wrong_answer']
        correct_tom_view = qa['correct_answer']
        if metric == "cosine":
            wrong_tom_view_emb = self.embedder.encode(wrong_tom_view)
            personx_view_emb = self.embedder.encode(correct_tom_view)
            model_response_emb = self.embedder.encode(model_response)
            similarity_wrong_tom_view = cosine_similarity(model_response_emb.reshape(1, -1), wrong_tom_view_emb.reshape(1, -1))[0][0]
            similarity_personx_view = cosine_similarity(model_response_emb.reshape(1, -1), personx_view_emb.reshape(1, -1))[0][0]
        else:
            raise NotImplementedError

        if similarity_wrong_tom_view >= similarity_personx_view:
            wrong_view_lexical_overlap = self.compute_f1(wrong_tom_view, model_response)
            return False, wrong_view_lexical_overlap
        else:
            personx_view_lexical_overlap = self.compute_f1(correct_tom_view, model_response)
            return True, personx_view_lexical_overlap

    def compute_f1(self, ground_truth, model_response):
        """
        Compute the F1 score between the ground truth and model response.

        Args:
            ground_truth (str): The ground truth text.
            model_response (str): The model's response text.

        Returns:
            float: The F1 score.
        """
        ground_truth = ground_truth.split()
        model_response = model_response.split()
        common = Counter(ground_truth) & Counter(model_response)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(model_response)
        recall = 1.0 * num_same / len(ground_truth)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def evaluate_mc_belief_q(self, qa, model_response):
        """
        Evaluate the multiple-choice version belief question.

        Args:
            qa (dict): The question and answer information.
            model_response (str): The model's response to the question.

        Returns:
            bool: True if the model's response matches the correct answer, False otherwise.
        """
        int_to_alphabet = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
        answer = int_to_alphabet[qa['correct_answer']]
        response = model_response.lower()

        if response.startswith("(" + answer + ")") or response.startswith(answer + ")") or response.startswith(answer + ".") or response.startswith(answer + ":") or response.startswith(answer + ",") or "({})".format(answer) in response or answer == response: # a) or a. or a or (a)
            return True
        else:
            return False
        
    
    def setup_fantom(self):
        """
        Flatten the dictionary and add short and full conversation context to each question.
        The result will be a list of questions and list of short or full inputs to be used as input for the models.
        """
        if self.args.aggregation_target == "conversation":
            assert self.args.conversation_input_type == "full", "The input type should have been the full conversation. It doesn't make sense to aggregate the scores over the full conversation when the input is not the full conversation"

        self.fantom_df_to_run = self.fantom_df

        total_num_q = 0
        for idx, _set in self.fantom_df_to_run.iterrows():
            total_num_q += len(_set['beliefQAs'])
            total_num_q += len(_set['answerabilityQAs_binary'])
            total_num_q += len(_set['infoAccessibilityQAs_binary'])
            if _set['factQA'] is not None:
                total_num_q += 1
            if _set['answerabilityQA_list'] is not None:
                total_num_q += 1
            if _set['infoAccessibilityQA_list'] is not None:
                total_num_q += 1

        inputs = []
        qas = []
        for idx, _set in self.fantom_df_to_run.iterrows():
            if self.args.conversation_input_type == "short":
                context = _set['short_context'].strip()
            elif self.args.conversation_input_type == "full":
                context = _set['full_context'].strip()
            else:
                raise NotImplementedError
            
            set_id = _set['set_id']
            fact_q = _set['factQA']['question']
            fact_a = _set['factQA']['correct_answer']

            # Fact Question
            _set['factQA']['context'] = context
            input_text = "{}\n\nQuestion: {}\nAnswer:".format(context, fact_q)
            _set['factQA']['input_text'] = input_text
            _set['factQA']['set_id'] = set_id
            qas.append(_set['factQA'])
            inputs.append(input_text)

            for _belief_qa in _set['beliefQAs']:
                # Belief Questions
                _belief_qa['context'] = context
                input_text = "{}\n\nQuestion: {}\nAnswer:".format(context, _belief_qa['question'])
                _belief_qa['input_text'] = input_text
                _belief_qa['set_id'] = set_id
                qas.append(_belief_qa)
                inputs.append(input_text)

                # Multiple Choice Belief Questions
                _mc_belief_qa = {**_belief_qa}
                choices_text, answer = self.set_beliefQA_multiple_choices(_mc_belief_qa)
                mc_question = "{}\n{}\n\nChoose an answer from above:".format(_belief_qa['question'], choices_text.strip())
                _mc_belief_qa['question'] = mc_question
                _mc_belief_qa['question_type'] = _mc_belief_qa['question_type'] + ":multiple-choice"
                _mc_belief_qa['choices_text'] = choices_text
                _mc_belief_qa['choices_list'] = choices_text.strip().split("\n")
                _mc_belief_qa['correct_answer'] = answer
                input_text = "{}\n\nQuestion: {}".format(context, mc_question)
                _mc_belief_qa['input_text'] = input_text
                qas.append(_mc_belief_qa)
                inputs.append(input_text)

            # Answerability List Questions
            _set['answerabilityQA_list']['fact_question'] = fact_q
            _set['answerabilityQA_list']['context'] = context
            input_text = "{}\n\nTarget: {}\nQuestion: {}\nAnswer:".format(context, fact_q, _set['answerabilityQA_list']['question'])
            _set['answerabilityQA_list']['input_text'] = input_text
            _set['answerabilityQA_list']['set_id'] = set_id
            if self.args.conversation_input_type == "full" and len(_set['answerabilityQA_list']['wrong_answer']) > 0:
                _set['answerabilityQA_list']['missed_info_accessibility'] = 'inaccessible'
            qas.append(_set['answerabilityQA_list'])
            inputs.append(input_text)

            # Answerability Binary Questions
            if self.args.conversation_input_type == "full":
                missed_info_accessibility_for_full = _set['answerabilityQAs_binary'][0]['missed_info_accessibility']
                for _info_accessibility_qa in _set['answerabilityQAs_binary']:
                    if _info_accessibility_qa['correct_answer'] != "yes":
                        missed_info_accessibility_for_full = 'inaccessible'

            for _answerability_qa in _set['answerabilityQAs_binary']:
                _answerability_qa['fact_question'] = fact_q
                _answerability_qa['context'] = context
                input_text = "{}\n\nTarget: {}\nQuestion: {} Answer yes or no.\nAnswer:".format(context, fact_q, _answerability_qa['question'])
                _answerability_qa['input_text'] = input_text
                _answerability_qa['set_id'] = set_id
                if self.args.conversation_input_type == "full":
                    _answerability_qa['missed_info_accessibility'] = missed_info_accessibility_for_full
                qas.append(_answerability_qa)
                inputs.append(input_text)

            # Info Accessibility List Questions
            _set['infoAccessibilityQA_list']['fact_question'] = fact_q
            _set['infoAccessibilityQA_list']['fact_answer'] = fact_a
            _set['infoAccessibilityQA_list']['context'] = context
            input_text = "{}\n\nInformation: {} {}\nQuestion: {}\nAnswer:".format(context, fact_q, fact_a, _set['infoAccessibilityQA_list']['question'])
            _set['infoAccessibilityQA_list']['input_text'] = input_text
            _set['infoAccessibilityQA_list']['set_id'] = set_id
            if self.args.conversation_input_type == "full" and len(_set['infoAccessibilityQA_list']['wrong_answer']) > 0:
                _set['infoAccessibilityQA_list']['missed_info_accessibility'] = 'inaccessible'
            qas.append(_set['infoAccessibilityQA_list'])
            inputs.append(input_text)

            # Info Accessibility Binary Questions
            if self.args.conversation_input_type == "full":
                missed_info_accessibility_for_full = _set['infoAccessibilityQAs_binary'][0]['missed_info_accessibility']
                for _info_accessibility_qa in _set['infoAccessibilityQAs_binary']:
                    if _info_accessibility_qa['correct_answer'] != "yes":
                        missed_info_accessibility_for_full = 'inaccessible'

            for _info_accessibility_qa in _set['infoAccessibilityQAs_binary']:
                _info_accessibility_qa['fact_question'] = fact_q
                _info_accessibility_qa['fact_answer'] = fact_a
                _info_accessibility_qa['context'] = context
                input_text = "{}\n\nInformation: {} {}\nQuestion: {} Answer yes or no.\nAnswer:".format(context, fact_q, fact_a, _info_accessibility_qa['question'])
                _info_accessibility_qa['input_text'] = input_text
                _info_accessibility_qa['set_id'] = set_id
                if self.args.conversation_input_type == "full":
                    _info_accessibility_qa['missed_info_accessibility'] = missed_info_accessibility_for_full
                qas.append(_info_accessibility_qa)
                inputs.append(input_text)

        self.inputs = inputs
        self.flattened_fantom = qas

    def set_beliefQA_multiple_choices(self, qa):
        if qa['question_type'].endswith(":inaccessible"):
            option_a = qa['wrong_answer']
            option_b = qa['correct_answer']
        else:
            option_a = qa['wrong_answer']
            option_b = qa['correct_answer']

        answer_goes_last = random.choice([True, False])
        if answer_goes_last:
            choices = [option_a, option_b]
            answer = 1
        else:
            choices = [option_b, option_a]
            answer = 0

        # option letters iterate over the alphabet
        option_letters = ["(" + chr(x) + ")" for x in range(ord('a'), len(choices) + ord('a'))]
        choices_text = ""
        for letter, option in zip(option_letters, choices):
            choices_text += "{} {}\n".format(letter, option)

        return choices_text, answer

    def parse_response(self, response):
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        elif "Choose an answer from above:" in response:
            response = response.split("Choose an answer from above:")[-1].strip()

        return response

if __name__ == "__main__":
    print("hellO")