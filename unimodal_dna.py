import torch
import os
import sys
import numpy as np
import random
from transformers import AutoTokenizer, AutoModel, set_seed
from torch import nn
from datasets import load_dataset
from dotenv import load_dotenv

from ModelModule import ModelModule


load_dotenv()
apitoken = os.getenv("API_KEY")

global device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DNAEncoder(ModelModule):
    def __init__(
            self,
            params: dict,
            run_name: str,
            d_enc: str = "zhihan1996/DNA_bert_6",
            d_tokenizer: str = "zhihan1996/DNA_bert_6",
            cache_dir: str | bool = False,
            ds_randomization: bool = False,
            augmentation: bool = False,
            hierarchical: bool = False,
        ):

        super().__init__(
            params=params,
            run_name=run_name,
            ds_randomization=ds_randomization,
            augmentation=augmentation,
            hierarchical=hierarchical
        )

        # Unique parameter for cropping of DNA sequence
        self.k = params["k"]

        # Load pretrained DNA-BERT Model
        if cache_dir:
            self.dna_encoder = AutoModel.from_pretrained(
                d_enc,
                token=apitoken,
                cache_dir=cache_dir,
            )
            self.dna_tokenizer = AutoTokenizer.from_pretrained(
                d_tokenizer,
                token=apitoken,
                cache_dir=cache_dir,
            )
        else:
            self.dna_encoder = AutoModel.from_pretrained(
                d_enc,
                token=apitoken,
            )
            self.dna_tokenizer = AutoTokenizer.from_pretrained(
                d_tokenizer,
                token=apitoken,
            )

        # Hardcoded output sizes of DINOV2_small and DNABERT_6
        d_enc_size = 768

        # Create projection for to make BERT features better suited for classification
        self.class_head = nn.Sequential(
            nn.Linear(d_enc_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128), 
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, [i for i in self.class_values.values()][0])
            ) # Maps embedding to species classes

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        # If the weights exist, we have to run from the checkpoint
        if os.path.exists(f"{run_name}/latest.pt"):
            self.start_from_checkpoint(f"{run_name}/")


    def collate_fn(self, batch, train: bool = False):
        """Custom collation function for dataloader, extract images from the batch and pad them with the largest width from the widest image.

        Args:
            batch(dict): The batch containing the data subset

        Returns:
            torch.tensor: The stacked tensor of the batch of images.
        """
        # Get labels based on the class index mapping for the current labeltype
        labels = [self.class_mapping[self.labeltype][i] for i in batch[self.labeltype]]
        
        # If augmentation is on and training mode is also on, random cropping of a 510-length sequence
        if self.augmentation and train:
            barcodes = self.kmer_crop(batch["dna_barcode"])
        else:
            barcodes = [" ".join([seq[i:i+self.k] for i in range(len(seq) - self.k + 1)]) for seq in batch["dna_barcode"]]
        
        # Tokenize our extracted barcode sequence
        tokenized_barcodes = self.dna_tokenizer(barcodes, return_tensors = 'pt', padding=True, truncation=True).to(device)

        return {"labels": torch.tensor(labels).long().to(device),
                "barcodes": tokenized_barcodes}


    def kmer_crop(self, barcodes, max_length=510, center: bool = False):    
        """Crop a random portion of kmers from the barcode.
        
        Args:
            barcodes (list): List of DNA barcodes
            max_length (integer): Maximum kmers to output, standard 510.
        
        Returns:
            List of kmer crop kmers.
        """
        if center:
            starts = [(len(sequence) - max_length) // 2 for sequence in barcodes]            
        else:
            starts = [np.random.randint(0, len(sequence) - max_length) if len(sequence) > max_length else 0 for sequence in barcodes]
        
        crops = [seq[start:start + max_length] for start, seq in zip(starts, barcodes)]
        kmer_crops = [" ".join([seq[i:i+self.k] for i in range(len(seq) - self.k + 1)]) for seq in crops]
        return kmer_crops


    def forward(self, batch):
        # CLS token as embedding
        embedding = self.dna_encoder(**batch["barcodes"]).last_hidden_state[:,0]
        logits = self.class_head(embedding)

        return logits


    def freezeencoders(self, until):
        """Function called during training loop, needs to be specified in each implementation for their respective encoders"""
        self.freeze_until(self.dna_encoder, until)

if __name__ == "__main__":
    # Set seed for everything
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

    # Create save location directory
    if not os.path.exists(f"{run_name}/"):
        os.mkdir(f"{run_name}/")
    
    # If cache needs to be used
    cache_dir = os.getenv("cache_dir", default=None)
    class_indices_path = os.getenv("class_indices", default="./class_indices/")

    print("Loading/Downloading training dataset.")

    # Train dataset
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
    eval_dataset = load_dataset("dataset.py", 
                                name="cropped_256_eval", 
                                split="validation", 
                                trust_remote_code=True,
                                token=apitoken,
                                cache_dir=cache_dir
    )
    eval_dataset = eval_dataset.with_format("torch", device=device)
    
    print("Generating/loading class indices.")
    
    # Initialize every single species as a valuen integer
    uniq_classes = set.union(set(train_dataset["class"]), set(train_dataset["class"]))
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

    uniq_species = set.union(set(train_dataset["species"]), set(train_dataset["species"]))
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

    print(f"Running model on {run_name}, with hierarchical: {hierarchical}, ds_rand: {ds_randomization}, augmentation: {augmentation}.")

    model = DNAEncoder(
        cache_dir=cache_dir,
        params=parameters,
        run_name = run_name,
        ds_randomization=ds_randomization,
        augmentation=augmentation,
        hierarchical=hierarchical,
    )

    model = model.to(device)
    model.train_loop(train_dataset, eval_dataset)
    model.save(f"{run_name}/")
    model.plot_metrics(save_path=f"{run_name}")
    