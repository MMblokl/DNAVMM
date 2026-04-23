import torch.nn as nn
import torch
import torchvision.transforms as T
from transformers import AutoModel, AutoTokenizer, set_seed, AutoImageProcessor
from datasets import load_dataset
import numpy as np
import random
import os
import sys
from dotenv import load_dotenv

from ModelModule import ModelModule

load_dotenv()
apitoken = os.getenv("API_KEY")

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DNAVMM(ModelModule):
    def __init__(
            self,
            params: dict,
            run_name: str,
            d_enc: str = "zhihan1996/DNA_bert_6",
            v_enc: str = "facebook/dinov2-small",
            d_tokenizer: str = "zhihan1996/DNA_bert_6",
            i_processor: str = 'facebook/dinov2-small',
            cache_dir: str | bool = False,
            ds_randomization: bool = False,
            augmentation: bool = False,
            hierarchical: bool = False,
        ):
        
        # Init all base functions
        super().__init__(
            params=params,
            run_name=run_name,
            ds_randomization=ds_randomization,
            augmentation=augmentation,
            hierarchical=hierarchical
        )

        # Whether to use pre-defined cache dir for parameters and data
        if cache_dir:
            self.dna_encoder = AutoModel.from_pretrained(
                d_enc,
                token=apitoken,
                cache_dir=cache_dir,
            )
            self.visual_encoder = AutoModel.from_pretrained(
                v_enc,
                token=apitoken,
                cache_dir=cache_dir,
            )
            self.dna_tokenizer = AutoTokenizer.from_pretrained(
                d_tokenizer,
                token=apitoken,
                cache_dir=cache_dir,
            )
            self.i_processor = AutoImageProcessor.from_pretrained(
                i_processor,
                token=apitoken,
                cache_dir=cache_dir,
            )
        else:
            self.dna_encoder = AutoModel.from_pretrained(
                d_enc,
                token=apitoken,
            )
            self.visual_encoder = AutoModel.from_pretrained(
                v_enc,
                token=apitoken,
            )
            self.dna_tokenizer = AutoTokenizer.from_pretrained(
                d_tokenizer,
                token=apitoken,
            )
            self.i_processor = AutoImageProcessor.from_pretrained(
                i_processor,
                token=apitoken,
            )
        
        # K parameter for cropping of k-mers
        self.k = params["k"]

        # Hardcoded output sizes of DINOV2_small and DNABERT_6
        d_enc_size = 768
        v_enc_size = 384

        # Augmentation Composer
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

        # Fully connected classification head
        self.class_head = nn.Sequential(
            nn.Linear(d_enc_size + v_enc_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128), 
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, [i for i in self.class_values.values()][0])
        )

        # Init optimizer and loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        # If the weights exist, we have to run from the checkpoint
        if os.path.exists(f"{run_name}/latest.pt"):
            self.start_from_checkpoint(f"{run_name}/")


    def freezeencoders(self, until):
        """Function called during training loop, needs to be specified in each implementation for their respective encoders"""
        self.freeze_until(self.visual_encoder, until)
        self.freeze_until(self.dna_encoder, until)


    def collate_fn(self, batch, train: bool = True):
        """Custom collation function for dataloader, extract images from the batch and pad them with the largest width from the widest image.

        Args:
            batch(dict): The batch containing the data subset
            train (boolean): Training mode yes/no.

        Returns:
            torch.tensor: The stacked tensor of the batch of images.
        """
        images = batch["image"]

        if self.augmentation and train:
            # Apply the image augmentation method to every image
            images = [self.augment(img) for img in images]
            barcodes = self.kmer_crop(batch["barcodes"])
        else:
            barcodes = [" ".join([seq[i:i+self.k] for i in range(len(seq) - self.k + 1)]) for seq in batch["dna_barcode"]]

        tokenized_barcodes = self.dna_tokenizer(barcodes, return_tensors = 'pt', padding=True, truncation=True).to(device)
        images = self.i_processor(images=images, return_tensors="pt").to(device)
        labels = [self.class_mapping[self.labeltype][i] for i in batch[self.labeltype]]
        
        return {"images": images,
                "labels": torch.tensor(labels).long().to(device),
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
        #v_embedding = self.visual_encoder(images).last_hidden_state.mean(dim=1)
        #d_embedding = self.dna_encoder(**dna).last_hidden_state.mean(dim=1)
        # CLS for embedding.
        v_embedding = self.visual_encoder(**batch["images"]).last_hidden_state[:,0]
        d_embedding = self.dna_encoder(**batch["barcodes"]).last_hidden_state[:,0]

        # Combine 
        feature_vec = torch.cat([v_embedding, d_embedding], dim=-1)

        # Pass through model.
        logits = self.class_head(feature_vec)

        return logits

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
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    class_indices_path = os.getenv("class_indices", default="./class_indices/")

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
    
    # Load the BIOSCAN5M validation dataset
    eval_dataset = load_dataset("dataset.py", 
                                name="cropped_256_eval", 
                                split="validation", 
                                trust_remote_code=True,
                                token=apitoken,
                                cache_dir=cache_dir
    )
    eval_dataset = eval_dataset.with_format("torch", device=device)
    
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

    model = DNAVMM(
        cache_dir=cache_dir,
        params=parameters,
        run_name = run_name,
        ds_randomization=ds_randomization,
        augmentation=augmentation,
        hierarchical=hierarchical,
    )
    model = model.to(device)
    model.train_loop(train_dataset, eval_dataset)
    model.save(f"{run_name}")
    model.plot_metrics(save_path=f"{run_name}")

