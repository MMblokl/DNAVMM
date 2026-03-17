# wget -P https://huggingface.co/datasets/bioscan-ml/BIOSCAN-5M/resolve/main/dataset.py
# pip install datasets==3.6.0

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
import requests
from transformers import AutoImageProcessor, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the DINOV2-small processor and model
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model = AutoModel.from_pretrained('facebook/dinov2-small')
model = model.to(device)
model.eval()

# Load the BIOSCAN5M dataset
test = load_dataset("dataset.py", name="cropped_256_train", split="train", trust_remote_code=True)
test = test.with_format("torch", device=device)

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
dataloader = DataLoader(test, batch_size=16, collate_fn=colate)

# Run the model
for batch in dataloader:
    inputs = processor(images=batch, return_tensors="pt") # Tokenizer
    outputs = model(**inputs) # Model output
    outputs.pooler_output # Max pooling layer output
    last_hidden_states = outputs.last_hidden_state # Hidden state
    last_hidden_states[:,0] # Classification token (CLS)
    last_hidden_states.mean(dim=1) # Mean pooling