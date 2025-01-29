import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MinimalCoconut
from dataset import CoconutDataset

def train(config):
    print(f"Training on device: {config.device}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    special_tokens = ['<|latent|>', '<|start-latent|>', '<|end-latent|>']
    num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    print(f"Added {num_added_tokens} special tokens to tokenizer")
    
    # Create base model and resize embeddings
    base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Create model
    model = MinimalCoconut(
        base_model,
        tokenizer.convert_tokens_to_ids('<|latent|>'),
        tokenizer.convert_tokens_to_ids('<|start-latent|>'),
        tokenizer.convert_tokens_to_ids('<|end-latent|>')
    )
    
    # Move model to device
    model.to(config.device)
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Prepare dataset
    train_dataset = CoconutDataset(config.train_path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Move batch to device
            batch = {k: v.to(config.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{config.num_epochs}, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if hasattr(config, 'checkpoint_dir'):
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            checkpoint_path = f"{config.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    from utils import Config
    
    config = Config({
        'model_name': 'gpt2',
        'train_path': 'data/sample.json',
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 3,
        'checkpoint_dir': 'checkpoints'
    })
    
    train(config)