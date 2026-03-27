from torch import nn
from transformers import AutoModel
from DNA_BERT_encoder import DNAEncoder

class DNAClassifier(nn.Module, num_classes=3):
    def __init__(self):
        super().__init__()
        self.encoder = DNAEncoder()
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, input_ids, attention_mask):
        emb = self.encoder(input_ids, attention_mask)
        return self.classifier(emb) 

if __name__ == "__main__":
    pass