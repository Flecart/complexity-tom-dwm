""" Script used to check the memorization of the model over a dataset. """
import os
from main.utils import queryLLM
import tiktoken
import numpy as np
from thefuzz import fuzz
import backoff

@backoff.on_exception(backoff.expo, Exception, max_time=120)
def queryGPT35Instruct(prompt: str):
    return queryLLM(prompt, './config.json', 'gpt-3.5-instruct')
__encoder = tiktoken.encoding_for_model("gpt-3.5-turbo-instruct")

def get_dataset(filename: str) -> tuple[list[str], list[str], list[str]]:

    # filename = os.path.join("data", "tomi", "test" + ".txt")
    with open(filename, "r") as f:
        s = f.read()
        return np.array(__encoder.encode(s))
    

def prompt_and_compare_tokenized(prompt: np.ndarray[int], start_index: int, offset: int):
    """ Check if memorization is present in the model.
    Args:
        prompt: The prompt to be used (correctly tokenized).
        start_index: The start index of the prompt.
        offset: The offset to be used.
    """

    text_prompt = __encoder.decode(prompt[start_index:start_index+offset])

    small_prompt = text_prompt
    response = queryGPT35Instruct(small_prompt)
    output = response["choices"][0]["text"]
    encoded_output = np.array(__encoder.encode(output))

    print(f"Prompt: {text_prompt}")
    print(f"Output: {output}")
    print(f"Match: {np.all(encoded_output == prompt[start_index+offset:start_index+offset+len(encoded_output)])}")
    
    return np.all(encoded_output == prompt[start_index+offset:start_index+offset+len(encoded_output)])


def compare_loop_tokenized():
    """ The compare function for the tokenized compare version

    It had a problem with ToMi because it seems we have better results if we split line by line instead of token by token.
    Sometimes it's just the anwer that is present in the response...
    """
    # ToMi has 438635 tokens
    dataset = get_dataset()
    offset = 400 # 400 tokens in prompt input
    print("Length of the dataset: ", len(dataset))

    number_matches = 0

    # - 30 because the output can be 30 tokens long
    max_len = len(dataset) - offset - 30
    for idx, i in enumerate(range(0, max_len, offset)):
        match = prompt_and_compare_tokenized(dataset, i, offset)
        if match:
            number_matches += 1

        if idx % 50 == 0:
            print(f"Offset: {idx}/{max_len // offset} - Number of matches: {number_matches}")

    with open("results.txt", "a+") as f:
        f.write(f"Offset: {offset}\n")
        f.write(f"Number of matches: {number_matches}\n")
        f.write(f"Total number of prompts: {max_len // offset}\n")
        f.write(f"Percentage of matches: {number_matches/ (max_len // offset) * 100}%\n")

def prompt_and_compare_line(prompt: list[str], start_index: int, offset: int):
    """ Check if memorization is present in the model.
    Args:
        prompt: The prompt to be used.
        start_index: The start index of the prompt.
        offset: The offset to be used.
    """
    text_prompt = "\n".join(prompt[start_index:start_index+offset])
    next_part = "\n".join(prompt[start_index+offset:])
    small_prompt = text_prompt + "\n" # this new line is important somehow.
    response = queryGPT35Instruct(small_prompt)
    output = response["choices"][0]["text"]

    print(f"Prompt: {small_prompt}")
    print(f"Original: -{next_part[:len(output)]}-")
    print(f"Output: -{output}-")
    print(f"Match: {next_part[:len(output)] == output}")

    return next_part[:len(output)] == output

def prompt_and_compare_line_fuzzy(prompt: list[str], start_index: int, offset: int):
    """ Check if memorization is present in the model.
    Args:
        prompt: The prompt to be used.
        start_index: The start index of the prompt.
        offset: The offset to be used.
    """
    text_prompt = "\n".join(prompt[start_index:start_index+offset])
    next_part = "\n".join(prompt[start_index+offset:])
    small_prompt = text_prompt + "\n" # this new line is important somehow.
    response = queryGPT35Instruct(small_prompt)
    output = response["choices"][0]["text"]

    print(f"Prompt: {small_prompt}")
    print(f"Original: -{next_part[:len(output)]}-")
    print(f"Output: -{output}-")
    print(f"Match: {next_part[:len(output)] == output}")

    return fuzz.ratio(next_part[:len(output)], output)/100

def main(filename: str):

    dataset = get_dataset(filename)
    dataset = __encoder.decode(dataset)
    dataset = dataset.strip() #.split("\n")
    dataset = re.split(r'\n|\\n', dataset)
    print(dataset)
    offset = 50 # 400 tokens in prompt input
    print("Length of the dataset: ", len(dataset))

    max_prompts = 100  # stops after checking 100 cases
    ctr = 0
    number_matches = 0
    fuzzy_matches = []
    max_len = len(dataset) - offset - 1
    for idx, i in enumerate(range(0, max_len, offset)):
        match = prompt_and_compare_line(dataset, i, offset)
        if match:
            number_matches += 1
            
        fuzzy_matches.append(prompt_and_compare_line_fuzzy(dataset, i, offset))

        if idx % 50 == 0:
            print(f"Offset: {idx}/{max_len // offset} - Number of matches: {number_matches}")
            
        if ctr >= max_prompts:
            break 
        else:
            ctr += 1

    with open("results.txt", "a+") as f:
        f.write(f"Offset: {offset}\n")
        f.write(f"Number of samples: {max_prompts}\n")
        f.write(f"Number of matches: {number_matches}\n")
        # f.write(f"Total number of prompts: {max_len // offset}\n")
        f.write(f"Percentage of matches: {number_matches/max_prompts}%\n")
        f.write(f"Fuzzy matches (ratio, std): {np.mean(fuzzy_matches), np.std(fuzzy_matches)}\n")

if __name__ == "__main__":
    import re
    # ToMi has 438635 tokens
    # filename = os.path.join("data", "tomi", "test" + ".txt")
    # filename = get_dataset("data/tomi/test.txt")
    # filename = os.path.join("data", "Adv-CSFB", "unexpected_contents.jsonl")
    # filename = os.path.join("data", "fantom", "fantom_v1.json")
    # filename = os.path.join("data", "mindgames", "test-00000-of-00001-7dfe9e22268ffc8b.csv")
    filename = os.path.join("data", "socialIQa", "dev.jsonl")
    main(filename)