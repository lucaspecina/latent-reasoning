import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MinimalCoconut
from dataset import CoconutDataset

def train(config):
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Add special tokens
    special_tokens = ['<|latent|>', '<|start-latent|>', '<|end-latent|>']
    tokenizer.add_tokens(special_tokens)
    
    # Create model
    model = MinimalCoconut(
        config.model_name,
        tokenizer.convert_tokens_to_ids('<|latent|>'),
        tokenizer.convert_tokens_to_ids('<|start-latent|>'),
        tokenizer.convert_tokens_to_ids('<|end-latent|>')
    )
    
    # Prepare dataset
    train_dataset = CoconutDataset(config.train_path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            optimizer.zero_grad()
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

if __name__ == "__main__":
    from utils import Config
    
    config = Config({
        'model_name': 'gpt2',
        'train_path': 'data/sample.json',
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 3
    })
    
    train(config)