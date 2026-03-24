from torch import nn
from transformers import AutoModel

class DNAEncoder(nn.Module):

    def __init__(self, model_name="zhihan1996/DNA_bert_6"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512)
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:,0,:]

        embedding = self.projection(cls)

        return embedding
    

if __name__ == "__main__":
    pass