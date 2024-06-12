# On the Notion of Complexity in ToM tasks

See main paper -> [TODO]

## Replication of the results.

You should create the files `config-azure.json` and `config.json` from the provided example files, namely `config-azure-example.json`, `config-example.json`. You should fill the needed keys.
Microsoft sponsorship is needed for the `gpt-4` and `gpt-3.5` models. If you don't have a microsoft sponsorship, you can adapt the code to work for standard openAI endpoints. It is possible that you need to make some modifications in the file located at `main/utils.py`.
The analysis of memorization just needs standard openAI key in `config.json` for `gpt-3.5-instruct`.

If you have everything set up. Just run `bash scripts/[model-name]` and wait some hours, and you will get the results.


## A simple pseudocode for the DWM

This section gives a simple example of how the partition function works:

```python

task = """
Jayden entered the master_bedroom.
Amelia entered the master_bedroom.
The strawberry is in the treasure_chest.
Amelia exited the master_bedroom.
Jayden moved the strawberry to the pantry.
Emma entered the master_bedroom.",
"""

question =  "Where will Amelia look for the strawberry?"

splits = partition_function(task, question)

# Should return a partition of the original task with the steps important to answer the question. Each split represents a state important to the question.
# splits = [
#     "Jayden entered the master_bedroom.\nAmelia entered the master_bedroom.",
#     "The strawberry is in the treasure_chest.",
#     "Amelia exited the master_bedroom.\nJayden moved the strawberry to the pantry.\nEmma entered the master_bedroom",
# ]

# ---------- Now we start with the DWM prompting ----------

# The initial prompt for the language model that asks for the completion.
prefix = """I give you a phrase of a dialogue between agents. I will reveal more parts of it later. At the end, I will give you a question you must answer. 
For each phrase, you must:
# 1. Write down a succinct description of what each agent knows about the environment and about the other agents. Keep the description short and do not produce redundant information. 
Here's the dialogue:
"""

assert len(splits) > 0

conversation = [{'role': 'system', 'content': prefix}]

# chat is the external API that returns the answer of the model, inputs a conversation, and outputs a completion string.
for i in range(len(splits)):
    conversation.append({'role': 'user', 'content': splits[i]})
    description = chat(conversation)
    conversation.append({'role': 'assistant', 'content': description})

# we have to finish the last iteration
final_ask = f"""This is the end of the dialogue. Now, this is a question for you to answer.

Question: Where will Amelia look for the strawberry?

Think step by step, answer the question with one word and provide the answer between <answer></answer> tags.
For example, reply with <answer>vase</answer>.
"""

conversation.append({'role': 'user', 'content': final_ask})
answer = chat(conversation)

# ---------- Now handle the LLMs answer ----------
```

In our case an approximation of the `partition_function` is implemented in the `split_context` function in `main.abstract_dataset`. The chat function is `queryLLM` in `main.utils` and generalizes a call to llms given the name and the config.