import re
import yaml
from main.tot_prompt import get_gpt

class Task():
    def __init__(self, database_name: str, kshot: int):
        self.database_name = database_name
        self.value_cache = {}
        self.kshot = kshot
        print(f"Loading task: {database_name} with {kshot} shot(s)...")

        with open(f"prompt/{database_name}/tot-{kshot}shot.txt", 'r') as f:
            self.prompt = f.read()

        with open(f"prompt/{database_name}/tot.yaml", 'r') as f:
            self.config = yaml.safe_load(f)

            self.propose_prompt = self.config['propose_prompt']
            self.propose_prompt_1 = self.config['propose_prompt_1']
            self.vote_prompt = self.config['vote_prompt']
            self.value_prompt = self.config['value_prompt']
            self.score_prompt = self.config['score_prompt']

    def prompt_wrap(self, x: str, y:str='') -> str:
        return self.prompt.format(input=x) + y
    
    def propose_prompt_wrap(self, x: str, y: str='') -> str:
        match int(self.kshot):
            case 0:
                return self.propose_prompt.format(input=x) + y
            case 1:
                return self.propose_prompt_1.format(input=x) + y
            case _:
                raise ValueError(f"kshot value {self.kshot} is not supported")

    def propose_outputs_unwrap(self, x: str, y: str, outputs: list, n_max_propose: int) -> list:
        confidence_to_value = {'certain': 1, 'high': 0.5, 'medium': 0.2, 'low': 0.1}  # TODO: ad hoc
        proposals_to_scores = {}
        for output in outputs:
            lines = output.split('\n')
            pattern = r'^(-)\. ([a-zA-Z]+) \((certain|high|medium|low)\).*$'
            for line in lines:
                match = re.match(pattern, line)
                if match:
                    parts = [match.group(1), match.group(2), match.group(3)]
                    proposal = parts[0].lower() + '. ' + parts[1].lower()
                    score = confidence_to_value.get(parts[2], 0)
                    proposals_to_scores[proposal] = proposals_to_scores.get(proposal, 0) + score
        
        proposals = sorted(proposals_to_scores.items(), key=lambda x: x[1], reverse=True)
        if n_max_propose != -1:
            proposals = proposals[:n_max_propose]
        proposals = [y + proposal[0] + '\n' for proposal in proposals]
        # self.cache_proposals[(x, y, n_max_propose)] = proposals
        return proposals
    
    # def evaluate(self, x: str, y: str, n_evaluate_sample: int) -> int:
    #     assert n_evaluate_sample == 1 # TODO: ad hoc
    #     count = {'sure': 0, 'maybe': 0, 'impossible': 0}
    #     for ans, data, status in zip(self.env.ans, self.env.data, self.env.status):
    #         if ans.count('_') >= 4: continue
    #         ans = ' '.join(ans.lower())
    #         line = f'{data}: {ans}'
    #         prompt = self.value_prompt.format(input=line)
    #         res = get_gpt()(prompt)[0]
    #         print(line)
    #         print(res)
    #         print()
    #         res = res.split('\n')[-1].strip()
    #         if res in count: count[res] += 1
    #     print(count)
    #     return count

    def vote_prompt_wrap(self, x: str, ys: list) -> str:
        prompt = self.vote_prompt.format(input=x)
        for i, y in enumerate(ys, 1):
            # y = y.replace('Plan:\n', '')
            # TODO: truncate the plan part?
            prompt += f'Choice {i}:\n{y}\n'
        return prompt
    
    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best choice is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        return vote_results
    
    def test_output(self, x: str, question: str, ys: list) -> str:
        prompt = self.score_prompt.format(problem=x, question=question, observations='\n'.join(ys))
        res = get_gpt()(prompt)[0]
        print(prompt)
        print("The proposed answer is: ", res)

        return res
