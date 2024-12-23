from transformers import BertModel, BertTokenizer
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(TextEncoder, self).__init__()
        print(f"Loading {model_name}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
    
    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1) 
