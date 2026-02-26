Each folder contains the code and outputs to run the pipeline for each model.

evaluation_template.txt and evaluation_template_pref.txt are the base prompt for every model except prometheus that has it's prompt.

You should first process the dataset, you then get a file "request.jsonl" then you can send API calls.

Llama should be run locally by downloading the weights.