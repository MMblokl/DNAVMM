# DNAVMM
Multi-modal DNA and vision species classification model for Multimodal Models in Ecology and Biodiversity.

# Command line arguments
In order to run the model in different modes, specific arguments have to supplied when running the scripts.

Dataset randomization during training: "ds_rand"
Data augmentation during training: "augment"
Hierarchical training scheme: "hierarchical"

Command examples:
- Using UV:
    - uv run python model.py augment ds_rand
- Using venv or conda:
    - python model.py hierarchical

# .env file envexample
There are 2 .env parameters one can set:
API_KEY; your huggingface API key
cache_dir; set to any directory to house the HF cache

- usage:
    - Rename "env_example" to ".env" and set the proper parameters as defined in the example

# unimodal_image.py
Obtain the dataset.py script from BIOSCAN5M to download the dataset: wget -P https://huggingface.co/datasets/bioscan-ml/BIOSCAN-5M/resolve/main/dataset.py

pip install datasets==3.6.0