import os
import sys
import torch

# Adding ViT-Prisma files and functions into current path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VIT_PRISMA_PATH = os.path.join(CURRENT_DIR, '..', 'ViT-Prisma', 'src')
if VIT_PRISMA_PATH not in sys.path:
    sys.path.insert(0, VIT_PRISMA_PATH)

# Depending on how many classes you are intending to test (Base: ImaangeNet-1K)
NUM_CLASSES = 1000 # TODO: Must change this if you want to test other datasets..


# Config path for using ViT-Prisma
MODEL_PATH = os.getenv("MODEL_PATH", "openai/clip-vit-base-patch32")
CACHE_DIR = os.getenv("HF_CACHE_DIR", "./cache")
DATA_PATH = os.getenv("DATA_PATH", "./data/imagenet_sketch")

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# NOTE: If you want more models, just add to this lsit.
SAE_PATHS = {
    8: "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-8-hook_resid_post-l1-1e-05",
    9: "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-9-hook_resid_post-l1-1e-05",
    10: "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-10-hook_resid_post-l1-1e-05",
    11: "Prisma-Multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-11-hook_resid_post-l1-1e-05"
}

# NOTE: Change the config if you want
DEFAULT_CONFIG = {
    'k': 1, 'gamma': 1.5, 'eta': 1.0, # The SparseAutoEncoder intervention parameters
    'spatial_selection': True, 
    'selection_config': { 'method': 'topk', 'top_k_percent': 0.4 }
}

# Image preprocessing. For CLIP-ViT/Openai
NORMALIZE_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORMALIZE_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGE_SIZE = (224, 224)

# For evaluation or maybe training
BATCH_SIZE = 32
SUBSET_SIZE = None