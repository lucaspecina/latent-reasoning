import torch
from typing import List, Tuple, Optional
import json
import os
from pathlib import Path

class CoconutDataGenerator:
    def __init__(
        self,
        tokenizer,
        latent_token: str = "<latent>",
        start_latent: str = "<start_latent>",
        end_latent: str = "<end_latent>",
        max_length: int = 512
    ):
        """
        Initialize the data generator for Coconut model training.
        
        Args:
            tokenizer: The tokenizer to use for encoding text
            latent_token: Token to represent latent variables
            start_latent: Token to mark the start of latent sequence
            end_latent: Token to mark the end of latent sequence
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.latent_token = latent_token
        self.start_latent = start_latent
        self.end_latent = end_latent
        self.max_length = max_length
        
        # Set up padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, 'eos_token'):
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # Add a new pad token if no eos token exists
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Add special tokens
        special_tokens = {
            "additional_special_tokens": [
                latent_token,
                start_latent,
                end_latent
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
    
    def create_sample(
        self,
        context: str,
        target: str,
        num_latents: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create a single training sample with context, latent tokens, and target.
        
        Args:
            context: The input context text
            target: The target completion text
            num_latents: Number of latent tokens to insert
            
        Returns:
            Tuple of (input_ids, attention_mask, labels)
        """
        # Create input sequence with latent tokens
        latent_sequence = f" {self.start_latent} " + \
                         f" {self.latent_token} " * num_latents + \
                         f" {self.end_latent} "
        
        input_text = context + latent_sequence + target
        
        # Tokenize
        encoded = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create labels (set to -100 for non-target tokens)
        labels = encoded["input_ids"].clone()
        
        # Find the end_latent token position
        end_latent_pos = (labels == self.tokenizer.convert_tokens_to_ids(self.end_latent))[0].nonzero()
        if len(end_latent_pos) > 0:
            end_pos = end_latent_pos[0].item()
            # Set labels before end of latent sequence to -100
            labels[0, :end_pos+1] = -100
        
        return (
            encoded["input_ids"],
            encoded["attention_mask"],
            labels
        )
    
    def create_batch_samples(
        self,
        samples: List[Tuple[str, str]],
        num_latents: int = 3,
        batch_size: Optional[int] = None
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Create multiple training samples.
        
        Args:
            samples: List of (context, target) pairs
            num_latents: Number of latent tokens per sample
            batch_size: Optional batch size for grouping samples
            
        Returns:
            List of (input_ids, attention_mask, labels) tuples
        """
        all_samples = []
        
        for context, target in samples:
            sample = self.create_sample(context, target, num_latents)
            all_samples.append(sample)
            
        if batch_size is not None:
            # Group samples into batches
            batched_samples = []
            for i in range(0, len(all_samples), batch_size):
                batch = all_samples[i:i + batch_size]
                if len(batch) == batch_size:  # Only keep full batches
                    input_ids = torch.cat([s[0] for s in batch])
                    attention_mask = torch.cat([s[1] for s in batch])
                    labels = torch.cat([s[2] for s in batch])
                    batched_samples.append((input_ids, attention_mask, labels))
            return batched_samples
            
        return all_samples

    def save_samples_to_json(
        self,
        samples: List[Tuple[str, str]],
        output_path: str,
        num_latents: int = 3
    ) -> None:
        """
        Generate and save samples to a JSON file in the format expected by CoconutDataset.
        
        Args:
            samples: List of (context, target) pairs
            output_path: Path to save the JSON file
            num_latents: Number of latent tokens per sample
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Convert samples to dataset format
        dataset = []
        for context, target in samples:
            sample = {
                "question": context,
                "steps": [""] * num_latents,  # Placeholder for latent tokens
                "answer": target
            }
            dataset.append(sample)
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(dataset)} samples to {output_path}")

def generate_example_samples() -> List[Tuple[str, str]]:
    """
    Generate some example training samples.
    
    Returns:
        List of (context, target) pairs
    """
    samples = [
        (
            "The capital of France is",
            "Paris. It is known as the City of Light."
        ),
        (
            "Python is a programming language that is",
            "known for its simplicity and readability."
        ),
        (
            "To make a cake, you need",
            "flour, eggs, sugar, and butter as basic ingredients."
        ),
        (
            "The three primary colors are",
            "red, blue, and yellow."
        )
    ]
    return samples

def generate_more_samples() -> List[Tuple[str, str]]:
    """
    Generate a larger set of training samples.
    """
    samples = [
        (
            "The capital of France is",
            "Paris. It is known as the City of Light."
        ),
        (
            "Python is a programming language that is",
            "known for its simplicity and readability."
        ),
        (
            "To make a cake, you need",
            "flour, eggs, sugar, and butter as basic ingredients."
        ),
        (
            "The three primary colors are",
            "red, blue, and yellow."
        ),
        (
            "The process of photosynthesis involves",
            "converting sunlight, water, and carbon dioxide into glucose and oxygen."
        ),
        (
            "The main components of a computer are",
            "CPU, RAM, storage, motherboard, and power supply."
        ),
        (
            "The water cycle consists of",
            "evaporation, condensation, precipitation, and collection."
        ),
        (
            "The human digestive system includes",
            "mouth, esophagus, stomach, small intestine, and large intestine."
        )
    ]
    return samples

if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    generator = CoconutDataGenerator(tokenizer)
    
    # Print tokenizer info
    print(f"Pad token: {tokenizer.pad_token}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    
    # Generate and save samples
    samples = generate_more_samples()
    
    # Save to JSON file
    output_path = "data/sample.json"
    generator.save_samples_to_json(samples, output_path)
    
    # Create and display some batches
    batch_samples = generator.create_batch_samples(
        samples,
        num_latents=3,
        batch_size=2
    )
    
    print(f"\nGenerated {len(batch_samples)} batches of samples")
    for i, (input_ids, attention_mask, labels) in enumerate(batch_samples):
        print(f"\nBatch {i+1}:")
        print(f"Input shape: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        print(f"Labels shape: {labels.shape}") 