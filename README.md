# DNAVMM
Multi-modal DNA and vision species classification model for Multimodal Models in Ecology and Biodiversity.

# Command line arguments
In order to run the model in different modes, specific arguments have to supplied when running the scripts.

First argument defines the run name, under which weights and metrics are saved

Dataset randomization during training: "ds_rand"
Data augmentation during training: "augment"
Hierarchical training scheme: "hierarchical"
Make the DNABERT self-attention input size 1024: "large_tokenizer", only works on unimodal_DNA and fusion model.

Command examples:
- Using UV:
    - uv run python model.py RUN_1 augment ds_rand large_tokenizer
- Using venv or conda:
    - python model.py RUN_2 hierarchical

# .env file envexample
There are 2 .env parameters one can set:
API_KEY; your huggingface API key
cache_dir; set to any directory to house the HF cache

- usage:
    - Rename "env_example" to ".env" and set the proper parameters as defined in the example

# unimodal_image.py
Obtain the dataset.py script from BIOSCAN5M to download the dataset: wget -P https://huggingface.co/datasets/bioscan-ml/BIOSCAN-5M/resolve/main/dataset.py

pip install datasets==3.6.0

# Model checkpointing
The model automatically loads from a given checkpoint every 5 epochs, and will load from weights given a specific run_name.
Every 5 epochs, model.weights will be replaced with the last weights to reduce memory usage.
! Make sure to not run the script again if the model has finished training completely, as this might override results.

# Class indices
The very first run of any model generated class_indices directory. This is a bit of a roundabout way to do this but it works fine.
This is then subsequently used by any other model while training, and will need to be deleted when training on a different dataset.