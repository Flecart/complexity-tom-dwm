from .base import Task
import re

class AdvCsfbTask(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.steps = 4
        self.stops = ['\n'] * 4

    # # Customize to your needs
    # def prompt_wrap(self, x: str, y:str='') -> str:
    #     return super().prompt_wrap(x, y)

    # def propose_prompt_wrap(self, x: str, y: str='') -> str:
    #     return super().propose_prompt_wrap(x, y)
    
    # def propose_outputs_unwrap(self, x: str, y: str, outputs: list, n_max_propose: int) -> list:
    #     return super().propose_outputs_unwrap(x, y, outputs, n_max_propose)
    
    # def evaluate(self, x: str, y: str, n_evaluate_sample: int) -> int:
    #     return super().evaluate(x, y, n_evaluate_sample)
    
    # def vote_prompt_wrap(self, x: str, ys: list) -> str:
    #     a =  super().vote_prompt_wrap(x, ys)
    #     return a
    
    # @staticmethod
    # def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
    #     return Task.vote_outputs_unwrap(vote_outputs, n_candidates)

    # def test_output(self, x: str, ys: list) -> str: 
    #     return super().test_output(x, ys)

    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best choice is .*(A|B|\d+).*" # aggiungere A|B perché non funziona in altri casi...
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                first = match.groups()[0]
                if first == 'A':
                    vote = 1
                elif first == 'B':
                    vote = 0
                elif first.isdigit():
                    vote = int(first) - 1
                else:
                    print(f'vote no match: {vote_output}')
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        return vote_results