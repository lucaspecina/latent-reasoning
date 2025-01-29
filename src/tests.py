import unittest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from utils import Config
from dataset import CoconutDataset
from model import MinimalCoconut

class TestCoconutSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up any resources needed for all tests"""
        print("\n=== Setting up test environment ===")
        cls.config_dict = {
            'model_name': 'gpt2',
            'train_path': 'data/sample.json',
            'batch_size': 4,
            'learning_rate': 1e-4,
            'num_epochs': 3
        }
        
        # Initialize config with device
        cls.config = Config(cls.config_dict)
        print(f"Using device: {cls.config.device}")
        
        print("Initializing tokenizer and model...")
        cls.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        cls.base_model = AutoModelForCausalLM.from_pretrained('gpt2')
        
        # Set padding token
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        print(f"Padding token set to: {cls.tokenizer.pad_token}")
        
        # Add special tokens
        cls.special_tokens = ['<|latent|>', '<|start-latent|>', '<|end-latent|>']
        num_added_tokens = cls.tokenizer.add_special_tokens({'additional_special_tokens': cls.special_tokens})
        print(f"Added {num_added_tokens} special tokens to tokenizer")
        
        # Resize model embeddings
        old_vocab_size = cls.base_model.get_input_embeddings().weight.shape[0]
        cls.base_model.resize_token_embeddings(len(cls.tokenizer))
        new_vocab_size = cls.base_model.get_input_embeddings().weight.shape[0]
        print(f"Resized model embeddings from {old_vocab_size} to {new_vocab_size}")
        
        # Store token IDs for later use
        cls.latent_id = cls.tokenizer.convert_tokens_to_ids('<|latent|>')
        cls.start_latent_id = cls.tokenizer.convert_tokens_to_ids('<|start-latent|>')
        cls.end_latent_id = cls.tokenizer.convert_tokens_to_ids('<|end-latent|>')
        print("Special token IDs:", {
            '<|latent|>': cls.latent_id,
            '<|start-latent|>': cls.start_latent_id,
            '<|end-latent|>': cls.end_latent_id
        })

    def create_model(self):
        """Helper to create and properly initialize model"""
        model = MinimalCoconut(
            self.base_model,
            self.latent_id,
            self.start_latent_id,
            self.end_latent_id
        )
        model.to(self.config.device)
        return model

    def test_config(self):
        """Test Config class functionality"""
        print("\n=== Testing Config ===")
        config = Config(self.config_dict)
        for key, value in self.config_dict.items():
            actual_value = getattr(config, key)
            print(f"Config {key}: expected={value}, actual={actual_value}")
            self.assertEqual(actual_value, value)

    def test_tokenizer(self):
        """Test tokenizer and special tokens"""
        print("\n=== Testing Tokenizer ===")
        
        # Test special tokens
        for token in self.special_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            print(f"Token {token}: ID={token_id}")
            self.assertNotEqual(token_id, self.tokenizer.unk_token_id, 
                              f"Token {token} was not properly added (got unk_token_id)")
        
        # Test encoding/decoding
        test_text = "What is 2+2?\n<|start-latent|><|latent|><|end-latent|>"
        encoded = self.tokenizer.encode(test_text)
        decoded = self.tokenizer.decode(encoded)
        print(f"\nEncoding test:\nOriginal: {test_text}\nDecoded: {decoded}")
        self.assertIn('<|latent|>', decoded)

    def test_dataset(self):
        """Test dataset loading and item retrieval"""
        print("\n=== Testing Dataset ===")
        
        try:
            dataset = CoconutDataset('data/sample.json', self.tokenizer)
        except FileNotFoundError:
            print("Creating sample dataset file...")
            import json
            sample_data = [{
                "question": "What is 2+2?",
                "steps": ["First add 2 and 2"],
                "answer": "4"
            }]
            os.makedirs('data', exist_ok=True)
            with open('data/sample.json', 'w') as f:
                json.dump(sample_data, f)
            dataset = CoconutDataset('data/sample.json', self.tokenizer)
        
        print(f"Dataset size: {len(dataset)} items")
        
        # Test sample item
        sample_item = dataset[0]
        print("\nSample item structure:")
        for key, value in sample_item.items():
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            
        # Decode sample
        decoded_text = self.tokenizer.decode(sample_item['input_ids'])
        print(f"\nDecoded sample:\n{decoded_text}")

    def test_model_init(self):
        """Test model initialization"""
        print("\n=== Testing Model Initialization ===")
        model = self.create_model()
        
        # Print model structure summary
        print("\nModel components:")
        print(f"Base model: {type(model.base_model).__name__}")
        print(f"Embedding size: {model.embedding.weight.shape}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    def test_model_forward(self):
        """Test model forward pass"""
        print("\n=== Testing Model Forward Pass ===")
        model = self.create_model()
        
        # Create sample input
        input_text = "What is 2+2?\n<|start-latent|><|latent|><|end-latent|>"
        print(f"\nInput text: {input_text}")
        
        inputs = self.tokenizer(input_text, return_tensors='pt')
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        print("\nInput tensors:")
        for key, value in inputs.items():
            print(f"{key}: shape={value.shape}")
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            print("\nModel outputs:")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: shape={value.shape}")
                else:
                    print(f"{key}: {type(value)}")

    def test_training_step(self):
        """Test a single training step"""
        print("\n=== Testing Training Step ===")
        model = self.create_model()
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create sample batch
        try:
            dataset = CoconutDataset('data/sample.json', self.tokenizer)
        except FileNotFoundError:
            print("Creating sample dataset file...")
            import json
            sample_data = [{
                "question": "What is 2+2?",
                "steps": ["First add 2 and 2"],
                "answer": "4"
            }]
            os.makedirs('data', exist_ok=True)
            with open('data/sample.json', 'w') as f:
                json.dump(sample_data, f)
            dataset = CoconutDataset('data/sample.json', self.tokenizer)
            
        sample_item = dataset[0]
        batch = {
            k: v.unsqueeze(0).to(self.config.device) 
            for k, v in sample_item.items()
        }
        
        print("\nBatch structure:")
        for key, value in batch.items():
            print(f"{key}: shape={value.shape}")
        
        # Test training step
        model.train()
        optimizer.zero_grad()
        
        print("\nExecuting forward pass...")
        outputs = model(**batch)
        loss = outputs.loss
        print(f"Loss: {loss.item():.4f}")
        
        print("Executing backward pass...")
        loss.backward()
        optimizer.step()
        
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        print("Training step completed successfully")

    def test_gpu_support(self):
        """Test GPU support if available"""
        print("\n=== Testing GPU Support ===")
        model = self.create_model()
        
        # Create sample input
        input_text = "What is 2+2?"
        inputs = self.tokenizer(input_text, return_tensors='pt')
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        # Check device placement
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Input device: {inputs['input_ids'].device}")
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            output_device = outputs.logits.device
            print(f"Output device: {output_device}")
            
            # Verify all tensors are on same device
            self.assertEqual(str(next(model.parameters()).device), str(self.config.device))
            self.assertEqual(str(inputs['input_ids'].device), str(self.config.device))
            self.assertEqual(str(output_device), str(self.config.device))

    def test_cuda_availability(self):
        """Test CUDA availability and configuration"""
        print("\n=== Testing CUDA Configuration ===")
        
        # Check CUDA availability
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        else:
            print("Running on CPU because:")
            if not hasattr(torch, 'cuda'):
                print("- PyTorch was built without CUDA support")
            elif not torch.cuda.is_available():
                print("- No CUDA-capable GPU found or CUDA drivers not installed")
            
        # Print PyTorch version and build info
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"PyTorch debug build: {torch.version.debug}")

if __name__ == '__main__':
    unittest.main(verbosity=2) 