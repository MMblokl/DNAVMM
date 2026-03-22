# wget -P https://huggingface.co/datasets/bioscan-ml/BIOSCAN-5M/resolve/main/dataset.py
# pip install datasets==3.6.0

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
import requests
from transformers import AutoImageProcessor, AutoModel, ViTImageProcessor, ViTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the DINOV2-small processor and model
processor_dinov2 = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model_dinov2 = AutoModel.from_pretrained('facebook/dinov2-small')
model_dinov2 = model_dinov2.to(device)

# Load the Vision transformer base processor and model
processor_vitb = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model_vitb = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
model_vitb = model_vitb.to(device)

# Load the BIOSCAN5M dataset
dataset = load_dataset("dataset.py", name="cropped_256_train", split="train", trust_remote_code=True)
dataset = dataset.with_format("torch", device=device)

# Check max size of the image, pad every image to have the same width as the image with the largest width
def colate(batch):
    """Custom collation function for dataloader, extract images from the batch and pad them with the largest width from the widest image.

    Args:
        batch(dict): The batch containing the data subset

    Returns:
        torch.tensor: The stacked tensor of the batch of images.
    """
    batch  = [i["image"] for i in batch]
    max_width = max(img.shape[-1] for img in batch)
    padded = [F.pad(img, (0, max_width-img.shape[-1], 0, 0)) for img in batch]
    return torch.stack(padded)

# Batch maker handler for the dataset
dataloader = DataLoader(dataset, batch_size=16, collate_fn=colate)

def extract_components(dataloader, processor, model, return_types
):
    """
    Extract components from the Hugging face model.

    Args:
        dataloader: PyTorch DataLoader containing the resized dataset
        processor: HuggingFace image processor
        model: HuggingFace model
        return_type: Tuple containing the types of components to extract from the model ("tokenizer", "max_pool", "hidden_state", "cls", "mean_pool")

    Returns:
    """
    # Set model to evaluation mode
    model.eval()

    # Create dict for the return types
    return_dict = {k: [] for k in return_types}

    # Run the pretrained model and extract the components
    for batch in dataloader:
        inputs = processor(images=batch, return_tensors="pt") # Tokenizer
        if "tokenizer" in return_types:
            return_dict["tokenizer"] = inputs

        outputs = model(**inputs) # Model output
        outputs.pooler_output # Max pooling layer output
        if "max_pool" in return_types:
            return_dict["max_pool"] = outputs.pooler_output

        last_hidden_states = outputs.last_hidden_state # Hidden state
        if "hidden_state" in return_types:
            return_dict["hidden_state"] = last_hidden_states

        last_hidden_states[:,0] # Classification token (CLS)
        if "cls" in return_types:
            return_dict["cls"] = last_hidden_states[:,0]

        last_hidden_states.mean(dim=1) # Mean pooling
        if "mean_pool" in return_types:
            return_dict["mean_pool"] = last_hidden_states.mean(dim=1)

        break
    return return_dict

# Extract components from the DINOV2 model
embeddings_dino = extract_components(
    dataloader,
    processor_dinov2,
    model_dinov2,
    return_types=("tokenizer", "max_pool", "hidden_state", "cls", "mean_pool")
)
print(embeddings_dino)