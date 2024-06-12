# On the Notion of Complexity in ToM tasks

See main paper -> [TODO]

## Replication of the results.

You should create the files `config-azure.json` and `config.json` from the provided example files, namely `config-azure-example.json`, `config-example.json`. You should fill the needed keys.
Microsoft sponsorship is needed for the `gpt-4` and `gpt-3.5` models. If you don't have a microsoft sponsorship, you can adapt the code to work for standard openAI endpoints. It is possible that you need to make some modifications in the file located at `main/utils.py`.
The analysis of memorization just needs standard openAI key in `config.json` for `gpt-3.5-instruct`.

If you have everything set up. Just run `bash scripts/[model-name]` and wait some hours, and you will get the results.