import torch
from torch import nn
import torchvision.transforms as T
from datasets import load_dataset
from transformers import AutoModel, set_seed, AutoImageProcessor
from dotenv import load_dotenv
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import sys

from ModelModule import ModelModule


load_dotenv()
apitoken =os.getenv("API_KEY")

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisualEncoder(ModelModule):
    def __init__(self,
                params: dict,
                run_name: str,
                v_enc: str = "facebook/dinov2-small",
                i_processor: str = "facebook/dinov2-small",
                cache_dir: str | bool = False,
                ds_randomization: bool = False,
                augmentation: bool = False,
                hierarchical: bool = False
                ):
        super().__init__(
            params=params,
            run_name=run_name,
            ds_randomization=ds_randomization,
            augmentation=augmentation,
            hierarchical=hierarchical
        )

        # Determine whether to use the pre-defined cache directory for the parameters and data
        if cache_dir:
            self.visual_encoder = AutoModel.from_pretrained(
                v_enc,
                token=apitoken,
                cache_dir=cache_dir
            )
            self.i_processor = AutoImageProcessor.from_pretrained(
                i_processor,
                token=apitoken,
                cache_dir=cache_dir
            )

        else:
            self.visual_encoder = AutoModel.from_pretrained(
                v_enc,
                token=apitoken
            )
            self.i_processor = AutoImageProcessor.from_pretrained(
                i_processor,
                token=apitoken
            )

        # Initialize the hardcoded output size of DINOV2_small
        v_enc_size = 384

        # Set the Fully connected classification head
        self.class_head = nn.Sequential(
            nn.Linear(v_enc_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128), 
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, [i for i in self.class_values.values()][0])
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        # Set the image augmentations
        self.augment = T.Compose([
            # Convert the image to float32
            T.ConvertImageDtype(torch.float32),

            # Randomly flip the image
            T.RandomHorizontalFlip(), # 50% chance to flip horizontal
            T.RandomVerticalFlip(), # 50% chance to flip vertical

            # Randomly apply gaussian noise to the image
            T.RandomApply(
                [T.v2.GaussianNoise(sigma=0.075)],
            p=0.5), # 50% to apply gaussian gaussian noise

            # Convert the image back to int8
            T.ConvertImageDtype(torch.uint8)
        ])

        if os.path.exists(f"{run_name}/latest.pt"):
            self.start_from_checkpoint(f"{run_name}/")

    def collate_fn(self, batch, train: bool = False):
        """Custom collation function for the dataset, extract images from the batch and process them with the AutoImageProcessor.

        Args:
            batch(dict): The batch containing the data subset
            train (boolean): Training mode yes/no.

        Returns:
            torch.tensor: The stacked tensor of the batch of images.
            labels: The stacked tensor of the corresponding labels of the batch of images.
        """
        images = batch["image"]

        # Determine whether to perform image augmentations on the training images
        # If we are in evaluation mode this is turned off
        if self.augmentation and train:
            # Apply the image augmentation method to every image
            images = [self.augment(img) for img in images]

        # Process the images with the DINOV2 image processor
        images = self.i_processor(images=images, return_tensors="pt").to(device)

        # Map the hierarchical labels to class integers
        labels = [self.class_mapping[self.labeltype][i] for i in batch[self.labeltype]]
        
        return {"images": images,
                "labels": torch.tensor(labels).long().to(device)}
    
    def forward(self, batch):
        # CLS for embedding
        embedding = self.visual_encoder(**batch["images"]).last_hidden_state[:,0]

        # Pass the embedding through classification head
        logits = self.class_head(embedding)

        return logits

    def freezeencoders(self, until):
        """Function called during training loop, needs to be specified in each implementation for their respective encoders"""
        self.freeze_until(self.visual_encoder, until)

    def visualize_augment(self, train_dataset, save_path, n_images):
        """Visualization of some image augmentation operations."""
        # Create the output directory for the image plots
        os.makedirs(save_path, exist_ok=True)

        # Loop through the first set of images of the train dataset
        for idx in range(n_images):
            # Get the image dictionary for the current image
            img_dict = train_dataset[idx]

            # Get the original training image
            original_img = img_dict["image"]

            # Augment training image
            augmented_img = self.augment(original_img)

            # Create the subplot to show the original and augmented image
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            # Original image
            axes[0].imshow(T.ToPILImage()(original_img.cpu()))
            axes[0].set_title("Original")
            axes[0].axis("off")
            # Augmented image
            axes[1].imshow(T.ToPILImage()(augmented_img.cpu()))
            axes[1].set_title("Augmented")
            axes[1].axis("off")

            # Plot and save the image
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"Augmented_Image_{idx}.png"))
            plt.close(fig)
                
if __name__ == "__main__":
    # Set the seed for everything to be deterministic
    seed = 202667
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    options = sys.argv[1:]
    run_name = options[0]
    # Enable hierarchical training, False for singular species level
    hierarchical = True if "hierarchical" in options else False

    # Enable dataset randomization, False for no randomization
    ds_randomization = True if "ds_rand" in options else False

    # Enable image augmentation, False for no image augmention
    augmentation = True if "augment" in options else False

    # Enable cache directory, None for no cache directory
    cache_dir = os.getenv("cache_dir", default=None)
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    class_indices_path = os.getenv("class_indices", default="./class_indices/")

    # Create save location directory
    if not os.path.exists(f"{run_name}/"):
        os.mkdir(f"{run_name}/")

    # Load the BIOSCAN5M train dataset
    train_dataset = load_dataset(
        "dataset.py",
        name="cropped_256_train", 
        split="train", 
        trust_remote_code=True,
        token=apitoken,
        cache_dir=cache_dir
        )
    train_dataset = train_dataset.with_format("torch", device=device)

    # Load the BIOSCAN5M validation dataset
    eval_dataset = load_dataset(
        "dataset.py", 
        name="cropped_256_eval", 
        split="validation", 
        trust_remote_code=True, 
        token=apitoken,
        cache_dir=cache_dir
        )
    eval_dataset = eval_dataset.with_format("torch", device=device)

    # Initialize the taxonomic labels for organisms in the dataset
    uniq_classes = set.union(set(train_dataset["class"]), set(train_dataset["class"])) # Class taxonomic level
    class_dict = {entry: i for i, entry in enumerate(uniq_classes)}
    n_class = len(uniq_classes)

    uniq_orders = set.union(set(train_dataset["order"]), set(eval_dataset["order"])) # Order taxonomic level
    order_dict = {entry: i for i, entry in enumerate(uniq_orders)}
    n_orders = len(uniq_orders)

    uniq_families = set.union(set(train_dataset["family"]), set(eval_dataset["family"])) # Family taxonomic level
    family_dict = {entry: i for i, entry in enumerate(uniq_families)}
    n_family = len(uniq_families)

    uniq_genus = set.union(set(train_dataset["genus"]), set(eval_dataset["genus"])) # Genus taxonomic level
    genus_dict = {entry: i for i, entry in enumerate(uniq_genus)}
    n_genus = len(uniq_genus)

    uniq_species = set.union(set(train_dataset["species"]), set(train_dataset["species"])) # Species taxonomic level
    species_dict = {entry: i for i, entry in enumerate(uniq_species)}
    n_species = len(uniq_species)
    
    # Set is fully non-deterministic, so we fix that by using mapping files to class indices
    if not os.path.exists(class_indices_path):
        os.mkdir(class_indices_path)
        np.save(f"{class_indices_path}class.npy", class_dict)
        np.save(f"{class_indices_path}order.npy", order_dict)
        np.save(f"{class_indices_path}family.npy", family_dict)
        np.save(f"{class_indices_path}genus.npy", genus_dict)
        np.save(f"{class_indices_path}species.npy", species_dict)
    else:
        class_dict = np.load(f"{class_indices_path}class.npy", allow_pickle=True).item()
        order_dict = np.load(f"{class_indices_path}order.npy", allow_pickle=True).item()
        family_dict = np.load(f"{class_indices_path}family.npy", allow_pickle=True).item()
        genus_dict = np.load(f"{class_indices_path}genus.npy", allow_pickle=True).item()
        species_dict = np.load(f"{class_indices_path}species.npy", allow_pickle=True).item()

    if hierarchical:
        # Initialize parameters to perform hierarchical training
        parameters = dict(
            lr = 5e-5,
            steps_per_epoch=200,
            batch_size=16,
            class_values = {
                "class": n_class,
                "order": n_orders,
                "family":n_family,
                "genus": n_genus,
                "species": n_species,
                },
            class_mapping = {
                "class": class_dict,
                "order": order_dict,
                "family": family_dict,
                "genus": genus_dict,
                "species": species_dict,
                },
            epoch_ordering = {
                "class": 2,
                "order": 4,
                "family": 10,
                "genus": 25,
                "species": 100
                }, # Number of epochs for each step.
            layer_freezing = {
                "class": None,
                "order": 1,
                "family": 4,
                "genus": 5,
                "species": 9,
                }
        )
    
    else:
        # Initialize the parameters to perform singular species training
        parameters = dict(
            lr = 5e-5,
            steps_per_epoch=200,
            batch_size=16,
            class_values = {
                "species": n_species,
                },
            class_mapping = {
                "species": species_dict,
                },
            epoch_ordering = {
                "species": 100
                }, # Number of epochs for each step.
            layer_freezing = {
                "species": None,
                }
        )

    # Set the model parameters and train the model
    model = VisualEncoder(
        cache_dir=cache_dir,
        params=parameters,
        run_name=run_name,
        ds_randomization=ds_randomization,
        augmentation=augmentation,
        hierarchical=hierarchical,
    )
    model = model.to(device)
    model.train_loop(train_dataset, eval_dataset)

    # Save the model weights obtained
    model.save(f"{run_name}")

    # Plot the metrics of the model
    model.plot_metrics(save_path=f"{run_name}")

    if augmentation:
        # Visualize the training images and augment examples
        model.visualize_augment(
            train_dataset=train_dataset,
            save_path=f"{run_name}/augment_plot",
            n_images=20
        )