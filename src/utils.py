class Config:
    def __init__(self, config_dict, force_gpu=False):
        for key, value in config_dict.items():
            setattr(self, key, value)
        
        # Set device automatically
        import torch
        if force_gpu and not torch.cuda.is_available():
            raise RuntimeError("GPU was forced but CUDA is not available")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')