import copy as cp
import tiktoken
import numpy as np
import re

def get_same_values(array: list[str]) -> list[int]:
    """Returns array of integers, where the first occurent of the element inside the array is always -1, 
    but if it occurred before, it is the index of the first occurrence of the element.
    """

    same_values = [-1] * len(array)
    for i, value in enumerate(array):
        for j, value2 in enumerate(array[:i]):
            if value == value2:
                same_values[i] = j
                break

    return same_values

def delete_tag(result: str, tag: str = "answer"):
    """deletes anything between <tag> and </tag

    >>> delete_tag("kept<answer>hello</answer>")
    "kept"
    """
    return re.sub(rf"<{tag}>.*</{tag}>", "", result)

def extract_tag(result: str, tag: str = "answer"):
    """extracts anything between <tag> and </tag>

    >>> extract_tag("kept<answer>hello</answer>")
    "hello"
    """
    regex = rf"<{tag}>(.*)</{tag}>"
    res = re.findall(regex, result)
    if len(res) == 0:
        regex = rf"<{tag}>(.*)<{tag}>"
        res = re.findall(regex, result) # For other models that don't close the tag
    
    if len(res) == 0:
        return ""
    return re.findall(regex, result)[0]

def add_comment(text: str):
    """
    Adds a #GPT# at the beginning of each line

    >>> add_comment("hello\nworld")
    "## hello\n## world"
    """
    return "\n".join([line for line in text.split("\n")])

def num_tokens_from_string(string: str, encoder: tiktoken.Encoding) -> int:
    token_count = len(encoder.encode(string))
    return token_count

def remove_same_comments(prompt: str, comments: list[str], tag: str) -> str:
    """Removes the comments that are the repeated in the prompt, and returns the new prompt.
    Keeps the first occurrence of the comment.
    """
    repeated_comments = np.array(get_same_values(comments))
    prompt_without_comments = cp.deepcopy(prompt)

    for i in range(len(comments)):
        if repeated_comments[i] != -1:
            prompt_without_comments = prompt_without_comments.replace(f"@comment-{i}@\n", "")
        else:
            prompt_without_comments = prompt_without_comments.replace(f"@comment-{i}@", comments[i])

        # print endlines as \n
    return prompt_without_comments

def reduce_context(prompt: str, max_tokens: int, tag: str, encoder: tiktoken.Encoding) -> str:
    """ 
    Remove redundant context to fix max_length
    """
    prompt_without_comments, comments = extract_annotations(prompt, tag)
    prompt_without_duplicates = remove_same_comments(prompt_without_comments, comments, tag)
    n_tok = num_tokens_from_string(prompt_without_duplicates, encoder)

    if n_tok <= max_tokens:
        return prompt_without_duplicates

    # ----------------------------------------------------------------
    # if we still have too many tokens, we try to remove the comments one by one until it's enough
    prompt_without_comments, comments = extract_annotations(prompt_without_duplicates, tag)
    for i in range(len(comments) + 1):
        new_prompt = integrate_annotations(prompt_without_comments, comments, i, tag)
        n_tok = num_tokens_from_string(new_prompt, encoder)
        print(f"Num tokens: {n_tok}")
        # print(new_prompt)
        if n_tok <= max_tokens:
            return new_prompt

    raise Exception(f"prompt is too long even without comments.")

def extract_annotations(prompt: str, tag: str = "#GPT#") -> tuple[str, list[str]]:
    """ Given a prompt, extracts all the annotations of a certain tag, in this case a tag is everything preceded by the 'tag' string and followed by a space character.

    Returns a tuple with the prompt with the annotations and a list of annotations.
    """
    all_prompt_lines = prompt.split("\n")
    prompt_without_comments = ""
    comments = []
    old_comments = 0

    for line in all_prompt_lines:
        if line.startswith(f"{tag}"):
            comments.append(line)
            prompt_without_comments += f"\n@comment-{old_comments}@"
            old_comments += 1
        else:
            prompt_without_comments += "\n" + line

    return prompt_without_comments, comments

def integrate_annotations(prompt: str, comments: list[str], remove_first: int, tag: str) -> str:
    """ Integrates the comments into the prompt, and removes the first 'remove_first' comments.
    """
    new_prompt = cp.deepcopy(prompt)
    for i in range(len(comments)):
        if i < remove_first:
            new_prompt = new_prompt.replace(f"@comment-{i}@", tag)
        else:
            new_prompt = new_prompt.replace(f"@comment-{i}@", comments[i])
    return new_prompt

def add_no_info_comments(prompt: str, tag: str) -> str:
    """ Inserts '#GPT# Nothing changes from the previous interaction.' in the prompt where there is no model information
    description.
    There is no information description when there are no lines starting with the tag between two normal lines

    Currently this function does not behave well, so it is not used.
    """
    all_prompt_lines = prompt.split("\n")
    prompt_with_comments = all_prompt_lines[0] + "\n"

    for i in range(1, len(all_prompt_lines)):
        if not all_prompt_lines[i-1].startswith(tag) and not all_prompt_lines[i].startswith(tag):
            prompt_with_comments += f"#GPT# Nothing changes from the previous interaction.\n"
        prompt_with_comments += all_prompt_lines[i] + "\n"

    return prompt_with_comments

if __name__ == "__main__":
    print(extract_tag("<answer>hello<answer>"))
    # Test get_same_values function
    def test_get_same_values():
        assert get_same_values(['a', 'b', 'a', 'c', 'b', 'c', 'd']) == [-1, -1, 0, -1, 1, 3, -1]

    def test_add_no_info_comments():
        prompt = """Tom: I bought a new car.
Alice: Oh cool, Tom! Sorry I need to leave for a moment, see you!
#GPT# Alice knows Tom has a new car
#GPT# Alice leaves.
Bob: See you Alice!
Bob: See you Alice!
"""

        expected = """Tom: I bought a new car.
#GPT# Nothing changes from the previous interaction.
Alice: Oh cool, Tom! Sorry I need to leave for a moment, see you!
#GPT# Alice knows Tom has a new car
#GPT# Alice leaves.
Bob: See you Alice!
#GPT# Nothing changes from the previous interaction.
Bob: See you Alice!
#GPT# Nothing changes from the previous interaction.

"""     
        assert add_no_info_comments(prompt, "#GPT#") == expected

    def test_extract_annotations():
        prompt = """Tom! Sorry I need to leave for a moment, see you!
#GPT# Alice knows Tom has a new car
#GPT# Alice leaves.
Bob: See you Alice!
Bob: See you Alice!
"""
        expeted_prompt = """
Tom! Sorry I need to leave for a moment, see you!
@comment-0@
@comment-1@
Bob: See you Alice!
Bob: See you Alice!
"""
        expected_comments = ["#GPT# Alice knows Tom has a new car", "#GPT# Alice leaves."]
        prompt_without_comments, comments = extract_annotations(prompt)

        assert prompt_without_comments == expeted_prompt
        for i in range(len(comments)):
            assert comments[i] == expected_comments[i]
    
    def test_remove_same_comments():
        assert remove_same_comments("Hello\n@comment-0@\nworld\n@comment-1@\n", ["Hello", "world"], "#GPT#") == "Hello\nHello\nworld\nworld\n"

    def test_remove_same_comments2():
        assert remove_same_comments("Hello\n@comment-0@\nworld\n@comment-1@\n", ["Hello", "Hello"], "#GPT#") == "Hello\nHello\nworld\n"

    # Test integrate_annotations function
    def test_integrate_annotations():
        assert integrate_annotations("\n@comment-0@\nworld!", ["#GPT# Hello"], 0, "#GPT#") == "\n#GPT# Hello\nworld!"

    def test_integrate_annotations_remove():
        assert integrate_annotations("\n@comment-0@\nworld!", ["#GPT# Hello"], 1, "#GPT#") == "\n#GPT#\nworld!"
    # reduce_context is not tested because its bad.

    test_get_same_values()
    test_add_no_info_comments()
    test_extract_annotations()
    test_integrate_annotations()
    test_integrate_annotations_remove()
    test_remove_same_comments()
    test_remove_same_comments2()