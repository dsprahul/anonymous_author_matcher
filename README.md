# Anonymous Author Matcher
Analyses the latent clues in writing styles and identifies the author if the database has content written by them.

### Dataset
* Posts and comments from 50 redditors are processed as per `dataset_maker.ipynb` and written to `./data`.
* Reusable ETL pipeline for training is in `redditors_comments_dataset.py`.

### Modeling Strategy 
* Train a sequence encoder model to distinguish whether the two input sequences are written by same author by setting it up as a binary classification task.

