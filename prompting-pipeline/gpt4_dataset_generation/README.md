# Dataset Generation with gpt4 API
This script is used to generate a dataset using the gpt4 API. 

**Disclaimer**: The current code is working for the LUSTER dataset, but can be easily adapted to other datasets by changing the processing functions in `utils.py` script.

The pipeline is as follows:

* `1-process_datasetname.py` Process your raw dataset to a format that can be used to generate gpt prompts
* `2-batch.py` Send the prompts in a **batch** to the gpt4 API to generate completions
* `3-get_batch.ipynb` Get the completions, process them to your wanted format and save them to a file
* The useful functions are in `utils.py`, and the generated files are saved in the `generated` folder
