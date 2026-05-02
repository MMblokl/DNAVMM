# DNAVMM
Multi-modal DNA and vision species classification model for Multimodal Models in Ecology and Biodiversity.

# .env file env_example
Optional parameters to set, but required if one want to run these on the leiden university clusters. Set the cache_dir to any /data/ directory to store huggingface cache files. API_KEY is the huggingface access token for faster downloading of the files and weights. class_indices can be kept as-is, unless you want to move this file anywhere else.

API_KEY; your huggingface API key

cache_dir; set to any directory to house the HF cache, only relavent if your home directory has storage quotas.

class_indices; The location of the class_indices directory. If running from the project repo directory directly, this does not need to be changed.

- usage:
    - After finalizing your settings, rename env_example to ".env" and make sure this file is present in the project directory when running.
    - These settings will only work if you run the script inside the project directory, so make sure to do this or else it will not function properly.

# Downloading required packages
There are multiple ways to install required packages, but UV is the easiest. The version numbers are VERY important due to BIOSCAN-5M requirements.
- uv:
    - Download uv on your global python installation:
        - pip install uv
        - pipx install uv
    - Restart your shell to get access to the uv command
    - Enter the project directory containing pyproject.toml
    - Download packages:
        - uv sync
    - Python has to be run with uv using the uv run python command:
        - uv run python fusion.py run_1 hierarchical ds_rand augment
    - **If this fails**, set UV_CACHE_DIR to some other location, might be required on the leiden university computing clusters with storage quotas on /home/.
        - Run export using any cache location with higher storage limits:
            - export UV_CACHE_DIR=/local/project/.cache/uv/ etc.
            - uv sync
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
In order to run the model with different settings, run the scripts with specific command line arguments:
- **IMPORTANT**
    - Very first argument should be the run directory, the location where all related files to the current one are saved; e.g. /data/user/run_1/
- Dataset randomization during training: "ds_rand"
- Data augmentation during training: "augment"
- Hierarchical training scheme: "hierarchical"

Command examples:
- Using UV:
    - uv run python model.py /data/run_1/ augment ds_rand
- Using venv or conda:
    - python model.py /data/run_2/ hierarchical ds_rand

# Creating metric graphs
To create a t-SNE visualization, simply run the following:
- After running a model using run_name as the model run:
    - python3 run_sne.py run_name

To create a training and validation loss plot of multiple runs of a model:
- Create a new dir "model_runs/" or "fusion_runs/" etc.
- Copy all model_metrics.npy files into the new directory and give them all a new name, for example: "hier_mod.npy", "base.npy", etc.
- Create the plots with the following:
    - python3 plotcreator.py model_runs


# Dataset
The dataset.py script is a script from the BIOSCAN-5M authors and downloads the dataset files. The current model is also only set up to handle this dataset as of writing.

**Important**
- This script is quite old and only works using dataset==3.6.0, using any other version will make the script stop functioning.

# Model checkpointing
The model automatically loads from a given checkpoint every 5 epochs, and will load from weights given a specific run_name.
Every 5 epochs, model.weights will be replaced with the last weights to reduce memory usage.

**Important**
- Make sure to not run the script again if the model has finished training completely, as this might override results.

# Class indices
The class_indices directory contains shared class indices for the BIOSCAN-5M dataset, and is generated again by any model script if removed. 
**Important**
- If removed, the link from indices to actuall classes dissapears, meaning you will have to re-train your from the start again.
