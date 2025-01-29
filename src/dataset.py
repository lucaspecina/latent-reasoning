import json
import torch
from torch.utils.data import Dataset

class CoconutDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format input with question and latent tokens
        input_text = (
            f"{item['question']}\n"
            "<|start-latent|>"
            "<|latent|>"
            "<|end-latent|>"
            f"{' '.join(item['steps'])}\n"
            f"### {item['answer']}"
        )
        
        # Tokenize
        encodings = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()
        }