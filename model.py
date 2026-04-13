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
    def __init__(self, d_enc, v_enc, d_tokenizer, i_processor, params):
        super(DNAVMM, self).__init__()
        
        
        self.visual_encoder = v_enc
        self.dna_encoder = d_enc
        self.dna_tokenizer = d_tokenizer
        self.i_processor = i_processor
        

        self.lr = params["lr"]
        self.n_classes = params["n_classes"]
        self.epochs = params["epochs"]
        self.steps_per_epoch = params["steps_per_epoch"]
        self.batch_size = params["batch_size"]
        self.k = params["k"]
    
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
            
            nn.Linear(128, self.n_classes)
            )
        self.dropout = nn.Dropout(0.1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        # Training metrics storage
        self.train_loss = np.zeros(shape=[self.epochs + 1, self.steps_per_epoch])
        self.eval_loss = np.zeros(shape=[self.epochs + 1, self.steps_per_epoch])
        self.train_acc = np.zeros(self.epochs)
        self.eval_acc = np.zeros(self.epochs)

    def collate_fn(self, batch):
        """Custom collation function for dataloader, extract images from the batch and pad them with the largest width from the widest image.

        Args:
            batch(dict): The batch containing the data subset

        Returns:
            torch.tensor: The stacked tensor of the batch of images.
        """
        images = batch["image"]
        images = self.i_processor(images=images, return_tensors="pt")
        labels = [species_dict[i] for i in batch["species"]]
        barcodes = [" ".join([seq[i:i+self.k] for i in range(len(seq) - self.k + 1)]) for seq in batch["dna_barcode"]]
        
        return {"images": images,
                "labels": torch.tensor(labels).long().to(device),
                "barcodes": barcodes}
   
    # Untested, probably works
    def forward(self, images, dna):
        #v_embedding = self.visual_encoder(images).last_hidden_state.mean(dim=1)
        #d_embedding = self.dna_encoder(**dna).last_hidden_state.mean(dim=1)
        # CLS for embedding.
        v_embedding = self.visual_encoder(**images).last_hidden_state[:,0]
        d_embedding = self.dna_encoder(**dna).last_hidden_state[:,0]

        # Combine 
        feature_vec = torch.cat([v_embedding, d_embedding], dim=-1)

        # Pass through model.
        feature_vec = self.dropout(feature_vec)
        logits = self.class_head(feature_vec)

        return logits
    
    # Might not be possible in same class, check RL implementations
    # Ideally, just doing model.train() will do it.
    def fit(self, train_dataset, eval_dataset):
        # Loop through each epoch
        for epoch in range(self.epochs):
            self.train() # Turn on dropouts
            # Shuffle the dataset at the start for more variable data training
            train_dataset = train_dataset.shuffle()
            # Print the current epoch
            print(f"Epoch {epoch+1}/{self.epochs}")
            
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
                
                # Tokenize each k-mer in the barcode
                tokenized_barcodes = self.dna_tokenizer(barcodes, return_tensors = 'pt', padding=True, truncation=True).to(device)

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

            # Calculate accuracy for the training dataset
            train_accuracy = train_correct / train_total
            
            # Turn off dropouts
            self.eval()

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

                    logits = self.forward(images=images)
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

            # Calculate accuracy for the validation dataset
            eval_accuracy = eval_prediction / eval_total

            # Save and print the metrics for the current epoch
            self.train_loss[epoch, :] = np.array(train_loss)
            self.train_acc[epoch] = np.array(train_accuracy)
            self.eval_loss[epoch, :] = np.array(eval_loss)
            self.eval_acc[epoch] = np.array(eval_accuracy)
            print("Training loss: ", self.train_loss.mean(axis=1)[epoch])
            print("Training accuracy:", train_accuracy)
            print("Validation loss: ", self.eval_loss.mean(axis=1)[epoch])
            print("Validation accuracy: ", eval_accuracy)
        
        # Print the training and evaluation loss log
        print("Training loss log: ", self.train_loss.mean(axis=1)[:-1])
        print("Validation loss log: ", self.eval_loss.mean(axis=1)[:-1])

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

    # Train dataset
    train_dataset = load_dataset(
        "dataset.py",
        name="cropped_256_train",
        split="train",
        trust_remote_code=True,
        token=apitoken,
        # cache_dir="/data/s4501888/hf/datasets"
    )
    train_dataset = train_dataset.with_format("torch", device=device)

    # Load the BIOSCAN5M validation dataset
    eval_dataset = load_dataset("dataset.py", 
                                name="cropped_256_eval", 
                                split="validation", 
                                trust_remote_code=True,
                                token=apitoken,
                                # cache_dir="/data/s4514998/hf/datasets"
    )
    eval_dataset = eval_dataset.with_format("torch", device=device)


    # Initialize every single species as a valuen integer
    uniq_species_train = set(train_dataset["species"])
    uniq_species_eval = set(train_dataset["species"])
    uniq_species = set.union(uniq_species_eval, uniq_species_train)
    n_classes = len(uniq_species)
    species_dict = {entry: i for i, entry in enumerate(uniq_species)}

    d_enc = AutoModel.from_pretrained(
        "zhihan1996/DNA_bert_6",
        token=apitoken,
        # cache_dir="/data/s4501888/hf/datasets"
    )
    v_enc = AutoModel.from_pretrained(
        "facebook/dinov2-small",
        token=apitoken,
        # cache_dir="/data/s4501888/hf/datasets"
    )
    d_tokenizer = AutoTokenizer.from_pretrained(
        "zhihan1996/DNA_bert_6",
        token=apitoken,
        # cache_dir="/data/s4501888/hf/datasets"
    )
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small',
                                                   token=apitoken,
                                            #    cache_dir="/data/s4514998/hf/datasets"
                                               )



    # Double size of the model for 1024 input size of DNA
    d_tokenizer.model_max_length = 1024
    d_enc.config.max_positional_embeddings = 1024
    d_enc.base_model.embeddings.position_ids = torch.arange(1024).expand((1,-1))
    d_enc.base_model.embeddings.token_type_ids = torch.zeros(1024).expand((1,-1))
    orig_pos_emb = d_enc.base_model.embeddings.position_embeddings.weight
    d_enc.base_model.embeddings.position_embeddings.weight = torch.nn.Parameter(torch.cat((orig_pos_emb, orig_pos_emb)))


    parameters = dict(
        lr = 1e-4,
        epochs=200,
        steps_per_epoch=200,
        batch_size=4,
        n_classes=n_classes,
        k=6,
    )


    model = DNAVMM(d_enc, v_enc, d_tokenizer, processor, params=parameters)
    model.to(device)

    model.fit(train_dataset, eval_dataset)
    model.save("/local/mmeb_s4501888/model.weights")

