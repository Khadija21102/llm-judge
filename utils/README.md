In this folder you can find various methods that can be useful for your pipelines.

- clean.py is called before the training to make sure the dataset is clean and do not contains any unexpected none values
- create_no_ref.py allow you to remove reference answers from the prompt of every possible file, it returns a new file
- distribution.py returns a new file with the same number of samples for each score
- extract_scores is called during the evaluation of models that returns 3d scores, it is called during the evaluation phase when 3-score is selected as the file type
- self_consistency.py creates a new file that contains the mean score of different runs of the same samples
- data_preprocessing.ipynb various modification on the data, creation of the files for the prompts
- bootstrap.py calculates the statistical significance of the ICC scores between 2 models
