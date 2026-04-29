# DNAVMM
Multi-modal DNA and vision species classification model for Multimodal Models in Ecology and Biodiversity.

# .env file envexample
Optional parameters to set, but required if one want to run these on the leiden university clusters. Set the cache_dir to any /data/ directory to store and huggingface cache files. API_KEY is the huggingface access token for faster downloading of the files and weights.

API_KEY; your huggingface API key

cache_dir; set to any directory to house the HF cache, only relavent if your home directory has storage quotas.

class_indices; The location of the class_indices directory. If running from the project repo directory directly, this does not need to be changed.

- usage:
    - After finalizing your settings, rename env_example to ".env" and make sure file is present in the project directory when running.
    - ! These settings will only work if you run the script inside the project directory, so make sure to do this or else it will not function properly.

# Downloading required packages
Multiple ways to install required packages, but UV is the easiest. The version numbers are VERY important due to BIOSCAN-1M requirements.
- uv:
    - Download uv on your global python installation:
        - pip install uv
    - Restart your shell to get access to the uv command
    - Enter the project directory containing pyproject.toml
    - Download packages:
        - uv sync
    - ! If uv doesnt work, set UV_CACHE_DIR to some other location, might be required on the leiden university computing clusters with the memory limitations.
    - Run export using any cache location with higher storage limits:
        - export UV_CACHE_DIR=/local/project/.cache/uv/ etc.
        - uv sync
    - ! Python has to be run with uv using the uv run python command:
        - uv run python fusion.py run_1 hierarchical ds_rand augment
- venv pip:
    - python -m venv .venv
    - source .venv/bin/activate
    - pip install -r requirements.txt
    - python fusion.py run_1 hierarchical ds_rand augment
- conda:
    - Create a conda environment
    - conda activate env_name
    - pip install -r requirements.txt
    - python fusion.py run_1 hierarchical ds_rand augment
- List of required packages:
    - torch == 2.7.0
    - datasets == 3.6.0
    - torchvision == 0.22.0
    - dotenv
    - matplotlib
    - transformers
    - scikit-learn

# Running the models
In order to run the model in different modes, specific arguments have to supplied when running the scripts.

Very first argument should be the run directory, the location where all related files to the current one are saved; e.g. /data/user/run_1/

Dataset randomization during training: "ds_rand"

Data augmentation during training: "augment"

Hierarchical training scheme: "hierarchical"

Command examples:
- Using UV:
    - uv run python model.py /data/run_1/ augment ds_rand large_tokenizer
- Using venv or conda:
    - python model.py /data/run_2/ hierarchical


# Dataset
The dataset.py script is a script made for getting the BIOSCAN-1M dataset, and the model is also set up to only take this dataset as of writing.

! This script is quite old and only works using dataset==3.6.0, using any other version will make the script stop functioning.

# Model checkpointing
The model automatically loads from a given checkpoint every 5 epochs, and will load from weights given a specific run_name.
Every 5 epochs, model.weights will be replaced with the last weights to reduce memory usage.

! Make sure to not run the script again if the model has finished training completely, as this might override results.

# Class indices
The very first run of any model generated class_indices directory if it is not yet present. These indices are for reproducibility and for making sure the model checkpoint doesnt start training using different indices.
