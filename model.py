import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, set_seed, AutoImageProcessor
from datasets import load_dataset
import numpy as np
import random
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt


load_dotenv()
apitoken = os.getenv("API_KEY")

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DNAVMM(nn.Module):
    def __init__(
            self,
            params: dict,
            d_enc: str = "zhihan1996/DNA_bert_6",
            v_enc: str = "facebook/dinov2-small",
            d_tokenizer: str = "zhihan1996/DNA_bert_6",
            i_processor: str = 'facebook/dinov2-small',
            cache_dir: str | bool = False,
        ):
        
        super(DNAVMM, self).__init__()

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
        
        # Use mode with larger self-attention matrix
        self.enlargen_tokenizer()

        # init parameters
        self.lr = params["lr"]

        self.class_values = params["class_values"]
        self.class_mapping = params["class_mapping"]
        self.epoch_ordering = params["epoch_ordering"]
        self.layer_freezing = params["layer_freezing"]

        self.steps_per_epoch = params["steps_per_epoch"]
        self.batch_size = params["batch_size"]
        self.k = params["k"]

        total_epochs = sum(self.epoch_ordering.values())
        
        # Hardcoded output sizes of DINOV2_small and DNABERT_6
        d_enc_size = 768
        v_enc_size = 384

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
            
            nn.Linear(128, self.class_values["class"])
            )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        # Training metrics storage
        self.train_loss = np.zeros(shape=[total_epochs + 1, self.steps_per_epoch])
        self.eval_loss = np.zeros(shape=[total_epochs + 1, self.steps_per_epoch])
        self.train_acc = np.zeros(total_epochs)
        self.eval_acc = np.zeros(total_epochs)

    def enlargen_tokenizer(self):
        # Double size of the model for 1024 input size of DNA
        self.dna_tokenizer.model_max_length = 1024
        self.dna_encoder.config.max_positional_embeddings = 1024
        self.dna_encoder.base_model.embeddings.position_ids = torch.arange(1024).expand((1,-1))
        self.dna_encoder.base_model.embeddings.token_type_ids = torch.zeros(1024).expand((1,-1))
        orig_pos_emb = self.dna_encoder.base_model.embeddings.position_embeddings.weight
        self.dna_encoder.base_model.embeddings.position_embeddings.weight = torch.nn.Parameter(torch.cat((orig_pos_emb, orig_pos_emb)))


    def collate_fn(self, batch):
        """Custom collation function for dataloader, extract images from the batch and pad them with the largest width from the widest image.

        Args:
            batch(dict): The batch containing the data subset

        Returns:
            torch.tensor: The stacked tensor of the batch of images.
        """
        images = batch["image"]
        images = self.i_processor(images=images, return_tensors="pt").to(device)
        labels = [self.class_mapping[self.labeltype][i] for i in batch[self.labeltype]]
        barcodes = [" ".join([seq[i:i+self.k] for i in range(len(seq) - self.k + 1)]) for seq in batch["dna_barcode"]]
        
        return {"images": images,
                "labels": torch.tensor(labels).long().to(device),
                "barcodes": barcodes}
   
    def forward(self, images, dna):
        #v_embedding = self.visual_encoder(images).last_hidden_state.mean(dim=1)
        #d_embedding = self.dna_encoder(**dna).last_hidden_state.mean(dim=1)
        # CLS for embedding.
        v_embedding = self.visual_encoder(**images).last_hidden_state[:,0]
        d_embedding = self.dna_encoder(**dna).last_hidden_state[:,0]

        # Combine 
        feature_vec = torch.cat([v_embedding, d_embedding], dim=-1)

        # Pass through model.
        logits = self.class_head(feature_vec)

        return logits
    
    def fit(self, train_dataset):
        """Fit the model on a shuffle of the dataset, once per epoch"""
        # Loop through each epoch
        self.train() # Turn on dropouts            
        prev = 0
        train_loss = []
        train_correct = 0
        train_total = 0

        for timestep, idx in enumerate(range(self.batch_size, len(train_dataset), self.batch_size)):
            self.optimizer.zero_grad() # Zero out previous grad

            # Collate data properly
            batch = self.collate_fn(train_dataset[prev:idx])

            # Single out data
            images = batch["images"]
            labels = batch["labels"]
            barcodes = batch["barcodes"]
            tokenized_barcodes = self.dna_tokenizer(barcodes, return_tensors = 'pt', padding=True, truncation=True).to(device)

            # Calculate loss
            logits = self.forward(images=images, dna=tokenized_barcodes)
            loss = self.criterion(logits, labels)

            prediction = torch.argmax(logits, dim=1)
            train_correct += (prediction == labels).sum().item()
            train_total += labels.size(0)

            # Backprop loss
            loss.backward()
            self.optimizer.step()
                
            # Save the loss to storage
            train_loss.append(loss.item())
            # Break training when steps are done
            if timestep == self.steps_per_epoch - 1:
                break
                
            # reset prev
            prev = idx
        return train_correct, train_total, train_loss

    def evaluate(self, eval_dataset):
        """Single evaluation run on dataset for each epoch"""
        # Initialize prev and the train loss, and the training metrics
        prev = 0
        eval_loss = []
        eval_prediction = 0
        eval_total = 0

        with torch.no_grad():  # Disable gradient computation for evaluation
            for timestep, idx in enumerate(range(self.batch_size, len(eval_dataset), self.batch_size)):
                # Collate the dataset to get the batches needed to evaluate the model
                batch = self.collate_fn(eval_dataset[prev:idx])
                        
                # Get the images and labels from the current batch
                images = batch["images"]
                labels = batch["labels"]
                barcodes = batch["barcodes"]
                tokenized_barcodes = self.dna_tokenizer(barcodes, return_tensors = 'pt', padding=True, truncation=True).to(device)
                

                logits = self.forward(images=images, dna=tokenized_barcodes)
                loss = self.criterion(logits, labels)

                # Save the loss
                eval_loss.append(loss.item())

                # Compute metrics
                prediction = torch.argmax(logits, dim=1)
                eval_prediction += (prediction == labels).sum().item()
                eval_total += labels.size(0)

                # Break the loop when the set number of training steps are reached
                if timestep == self.steps_per_epoch - 1:
                    break
                    
                # Reset prev
                prev = idx
        return eval_prediction, eval_total, eval_loss
    
    def freeze_until(self, model, until):
        """Freezes weights of given model given the value. If until=1, first 2 layers are frozen."""
        for i, layer in enumerate(model.encoder.layer):
            # If the index is withing the to be frozen layers
            if i <= until:
                for param in layer.parameters():
                    param.required_grad = False

    def update_optimizer(self):
        """Updates optimizer to only use trainable model params"""
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.lr)

    def train_loop(self, train_dataset, eval_dataset):
        # Loop through class label types
        for labeltype in self.class_values.keys():
            # Current type of class/label
            self.labeltype = labeltype
            epochs = self.epoch_ordering[labeltype] # Get number of epochs for the label
            
            until = self.layer_freezing[self.labeltype] # How many layers to freeze
            if until: # If None, no freeze.
                # Freeze the bottom n layers of both encoder
                self.freeze_until(self.dna_encoder, until)
                self.freeze_until(self.visual_encoder, until)

                # Make sure optim only optimizes on non-frozen weights
                self.update_optimizer()

            # Replace final class head layer with a new output layer matching the number of classes
            self.class_head[8] = nn.Linear(128, self.class_values[self.labeltype]).to(device)

            # Epoch loop
            for epoch in range(epochs):
                # Shuffle dataset
                train_dataset = train_dataset.shuffle()
                print(f"Epoch {epoch+1}/{epochs} for {labeltype}.")

                # Train on the dataset for the entire epoch
                train_correct, train_total, train_loss = self.fit(train_dataset=train_dataset)

                train_accuracy = train_correct / train_total
                # Save and print the metrics for the current epoch
                self.train_loss[epoch, :] = np.array(train_loss)
                self.train_acc[epoch] = np.array(train_accuracy)
                print("Training loss: ", self.train_loss.mean(axis=1)[epoch])
                print("Training accuracy:", train_accuracy)

        print("Training loss log: ", self.train_loss.mean(axis=1)[:-1])

        # for epoch in range(self.epochs):
            
        #     # Shuffle the dataset at the start for more variable data training
        #     train_dataset = train_dataset.shuffle()
        #     print(f"Epoch {epoch+1}/{self.epochs}")

        #     # Fit model on current shuffle
        #     train_correct, train_total, train_loss = self.fit(train_dataset=train_dataset)
        #     # Calculate accuracy for the training dataset
        #     train_accuracy = train_correct / train_total
            
        #     # Save and print the metrics for the current epoch
        #     self.train_loss[epoch, :] = np.array(train_loss)
        #     self.train_acc[epoch] = np.array(train_accuracy)
        #     print("Training loss: ", self.train_loss.mean(axis=1)[epoch])
        #     print("Training accuracy:", train_accuracy)

        #     # Valiation
        #     # eval_prediction, eval_total, eval_loss = self.evaluate(eval_dataset=eval_dataset)
        #     # Calculate accuracy for the validation dataset
        #     # eval_accuracy = eval_prediction / eval_total

        #     # self.eval_loss[epoch, :] = np.array(eval_loss)
        #     # self.eval_acc[epoch] = np.array(eval_accuracy)
        #     # print("Validation loss: ", self.eval_loss.mean(axis=1)[epoch])
        #     # print("Validation accuracy: ", eval_accuracy)
        
        # # Print the training and evaluation loss log
        # print("Training loss log: ", self.train_loss.mean(axis=1)[:-1])
        # # print("Validation loss log: ", self.eval_loss.mean(axis=1)[:-1])

    def plot_metrics(self, save_path):
        # Set the number of epochs used for the plots
        epochs = range(1, self.epochs + 1)

        # Plot the loss metrics for the model training and validation and save the figure
        fig, ax = plt.subplots()
        ax.plot(epochs, self.train_loss.mean(axis=1)[:-1], label="Train Loss")
        ax.plot(epochs, self.eval_loss.mean(axis=1)[:-1], label="Validation Loss")
        ax.set(xlabel="Epochs", ylabel="Loss", title="Training vs Validation Loss")
        ax.legend()
        fig.savefig(f"{save_path}_loss.png")

        # Plot the accuracy metrics for the model training and validation and save the figure
        fig, ax = plt.subplots()
        ax.plot(epochs, self.train_acc, label="Train Accuracy")
        ax.plot(epochs, self.eval_acc, label="Validation Accuracy")
        ax.set(xlabel="Epochs", ylabel="Accuracy", title="Training vs Validation Accuracy")
        ax.legend()
        fig.savefig(f"{save_path}_accuracy.png")
  
    def save(self, path):
        # Save the weights
        torch.save(self.state_dict(), path)

        # Save the metrics in one file
        metrics = {"train_loss": self.train_loss,
                   "train_acc": self.train_acc,
                   "eval_loss" : self.eval_loss,
                   "eval_acc": self.eval_acc,
                   "epochs": self.epochs
        }
        np.save(path + "_cls_species.npy", metrics)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))

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

    cache_dir = "/data/s4501888/hf/datasets"

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
    
    # For singlestage, add only species, and only those values for each other param.
    parameters = dict(
        lr = 5e-5,
        steps_per_epoch=200,
        batch_size=4,
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
            "order": 3,
            "family": 10,
            "genus": 20,
            "species": 200
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
    
    # Uncomment to enable SINGLE level
    # parameters = dict(
    #     lr = 5e-5,
    #     steps_per_epoch=200,
    #     batch_size=4,
    #     class_values = {
    #         "species": n_species,
    #         },
    #     class_mapping = {
    #         "species": species_dict,
    #         },
    #     epoch_ordering = {
    #         "species": 200
    #         }, # Number of epochs for each step.
    #     layer_freezing = {
    #         "species": None,
    #         },
    #     k=6,
    # )


    model = DNAVMM(
        cache_dir=cache_dir,
        params=parameters,
    )
    model = model.to(device)
    model.train_loop(train_dataset, eval_dataset)
    model.save("/local/mmeb_s4501888/model.weights")

