import torch
from datasets import load_dataset
from transformers import set_seed
from dotenv import load_dotenv
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys

from unimodal_image import VisualEncoder
from unimodal_dna import DNAEncoder
from fusion import DNAVMM

load_dotenv()
apitoken =os.getenv("API_KEY")

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TSNE_Visualiser:
    def __init__(self, model):
        self.model = model # Set the model class (VisualEncoder, DNAEncoder or DNAVMM)

    def embedding_extraction(self, batch):
        """
        Extract the specific modality embeddings out of either the unimodal or multimodal model.

        Args:
            batch(dict): The batch containing the data subset

        Returns: 
            torch.Tensor: the modality CLS for embedding
        """
        # Extract the Visual Encoder embedding
        if hasattr(self.model, "visual_encoder") and not hasattr(self.model, "dna_encoder"):
            embedding = self.model.visual_encoder(**batch["images"]).last_hidden_state[:, 0]

        # Extract the DNA Encoder embedding
        elif hasattr(self.model, "dna_encoder") and not hasattr(self.model, "visual_encoder"):
            embedding = self.model.dna_encoder(**batch["barcodes"]).last_hidden_state[:, 0]

        else:
            # Extract the Multimodal combined feature embedding
            v_embedding = self.model.visual_encoder(**batch["images"]).last_hidden_state[:, 0]
            d_embedding = self.model.dna_encoder(**batch["barcodes"]).last_hidden_state[:, 0]
            embedding = torch.cat([v_embedding, d_embedding], dim=-1)

        return embedding

    def collate_fn(self, batch, train: bool = True):
        """Custom collation function for dataloader, extract images from the batch and pad them with the largest width from the widest image.

        Args:
            batch(dict): The batch containing the data subset
            train (boolean): Training mode yes/no.

        Returns:
            torch.tensor: The stacked tensor of the batch of images.
        """
        images = batch["image"]

        if self.model.augmentation and train:
            # Apply the image augmentation method to every image
            images = [self.model.augment(img) for img in images]
            barcodes = self.model.kmer_crop(batch["dna_barcode"])
        else:
            barcodes = [" ".join([seq[i:i+self.model.k] for i in range(len(seq) - self.model.k + 1)]) for seq in batch["dna_barcode"]]

        tokenized_barcodes = self.model.dna_tokenizer(barcodes, return_tensors = 'pt', padding=True, truncation=True).to(device)
        images = self.model.i_processor(images=images, return_tensors="pt").to(device)
        labels = [self.model.class_mapping[self.model.labeltype][i] for i in batch[self.model.labeltype]]
        
        return {"images": images,
                "labels": labels,
                "barcodes": tokenized_barcodes}

    def run_tsne(self, dataset, save_path, max_timesteps=200, labelgroup="species"):
        """
        Running the t-SNE on the embedding extracted from the validation dataset, using a limited number of batches
        
        Args:

        Single evaluation run on the dataset for a limited number of batches
        """
        self.model.eval() # Set to evaluation mode
        # Initialize list to save the embedding and labels p
        embedding_list = []
        label_list = []

        prev = 0

        self.model.labeltype = labelgroup

        with torch.no_grad(): # Disable gradient computation for the t-sne on the validation dataset
            for timestep, idx in enumerate(range(self.model.batch_size, len(dataset), self.model.batch_size)):
                print(timestep)

                # Collate the dataset to get the batches needed to evaluate the model
                batch = self.collate_fn(dataset[prev:idx])

                # Extract the embeddings
                embedding = self.embedding_extraction(batch)

                # Add the embedding and labels for the current batch to their respective list
                embedding_list.append(embedding.cpu())
                label_list.append(batch["labels"])

                # Break the validation loop when the set number of training steps are reached
                if timestep == max_timesteps:
                    break

                # Set the previous index as the current index value for the next batch
                prev = idx

        # Convert the batch lists to tensor and then to NumPy for t-SNE
        embedding_list = torch.cat(embedding_list).numpy()
        label_list = [ x for xs in label_list for x in xs ]

        # Create the t-SNE model and fit it on the embeddings
        tsne = TSNE(n_components=2)
        tsne_fit = tsne.fit_transform(embedding_list)
        
        # Plot the t-SNE
        plt.figure(figsize=(6, 6))
        plt.scatter(tsne_fit[:, 0], tsne_fit[:, 1], c=label_list, s=10, cmap="tab20")
        plt.axis("off")
        plt.title(f"t-SNE {self.model.run_name} {self.model.labeltype}")
        plt.savefig(f"{save_path}_{self.model.labeltype}_tsne.png", bbox_inches="tight")
        plt.close()

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

    # Enable the command line argument to pass to the script
    options = sys.argv[1:]
    # Set the run name to use for the t-sne plot
    run_name = options[0]

    # Enable hierarchical training, False for singular species level
    hierarchical = True if "hierarchical" in options else False

    # Enable dataset randomization, False for no randomization
    ds_randomization = True if "ds_rand" in options else False

    # Enable image augmentation, False for no image augmention
    augmentation = True if "augment" in options else False

    # Enable cache directory, None for no cache directory
    cache_dir = os.getenv("cache_dir", default=None)
    class_indices_path = os.getenv("class_indices", default="./class_indices/")

    # Enable the model class to use as the weights and architecture of t-SNE
    if "image" in options:
        model_class = VisualEncoder
    elif "dna" in options:
        model_class = DNAEncoder
    else:
        model_class = DNAVMM
    print(f"Weights obtained from: {model_class}")

    # Load the BIOSCAN5M train dataset
    print("Loading/Downloading training dataset.")
    train_dataset = load_dataset(
        "dataset.py",
        name="cropped_256_train", 
        split="train", 
        trust_remote_code=True,
        token=apitoken,
        cache_dir=cache_dir
        )
    train_dataset = train_dataset.with_format("torch", device=device)

    print("Loading/Downloading validation dataset.")
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

    print("Generating/loading class indices.")

    # Initialize every single species as a valuen integer
    uniq_classes = set.union(set(train_dataset["class"]), set(eval_dataset["class"]))
    class_dict = {entry: i for i, entry in enumerate(uniq_classes)}
    n_class = len(uniq_classes)

    uniq_orders = set.union(set(train_dataset["order"]), set(eval_dataset["order"]))
    order_dict = {entry: i for i, entry in enumerate(uniq_orders)}
    n_orders = len(uniq_orders)

    uniq_families = set.union(set(train_dataset["family"]), set(eval_dataset["family"]))
    family_dict = {entry: i for i, entry in enumerate(uniq_families)}
    n_family = len(uniq_families)

    uniq_genus = set.union(set(train_dataset["genus"]), set(eval_dataset["genus"]))
    genus_dict = {entry: i for i, entry in enumerate(uniq_genus)}
    n_genus = len(uniq_genus)

    uniq_species = set.union(set(train_dataset["species"]), set(eval_dataset["species"]))
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
                },
            k=6,
        )
    else:
        parameters = dict(
            lr = 5e-5,
            steps_per_epoch=200,
            batch_size=16,
            class_values = {
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
                "species": 100
                }, # Number of epochs for each step.
            layer_freezing = {
                "species": None,
                },
            k=6,
        )


    print(f"Running t-SNE on {run_name}, with hierarchical: {hierarchical}, ds_rand: {ds_randomization}, augmentation: {augmentation}.")

    # Set the model parameters and train the model
    model = model_class(
        cache_dir=cache_dir,
        params=parameters,
        run_name=run_name,
        ds_randomization=ds_randomization,
        augmentation=augmentation,
        hierarchical=hierarchical,
    )
    model = model.to(device)

    # Load the model weights
    model.load(f"{run_name}/latest.pt")

    # Run the t-SNE
    tsne = TSNE_Visualiser(model)
    print("Running the t-SNE.")
    tsne.run_tsne(
    dataset=eval_dataset,
    save_path=f"{run_name}/",
    labelgroup="genus"
    )
    tsne.run_tsne(
    dataset=eval_dataset,
    save_path=f"{run_name}/",
    labelgroup="species"
    )