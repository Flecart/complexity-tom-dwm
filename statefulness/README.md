This section contains the small web application used to manually label the data.
The data is all structured in this format:

1. Prompt: the original task.
2. Question: our *obj*, the entity we are asking of.
3. Answer: the ground truth of our dataset.
4. states: the number of states that align with the question.
5. highlights: the human labelled highlights of the parts of the task that are useful in aswering the question.

If the states is -1, it is not labelled or invalid.
If the number of states is different from the length of the highlights array, it is then invalid.

We created the datasets with the script `to_json.py`. Run with `poetry run python3 -m  statefulness.to_json` from the root of the directory.
We moved the created dataset and created our final values with `copy_state_data.py`.

You can run those scripts from the root of the project with:

`python3 -m statefulness.to_json` and similar.


## App

After you have created the empty datasets. Just `cd app && bash start.sh` and the small demo app for labelling will start. Everything will be local.