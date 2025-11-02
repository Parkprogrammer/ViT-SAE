import torch

from huggingface_hub import hf_hub_download, list_repo_files
from . import config
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.sae.sae import SparseAutoencoder
from vit_prisma.dataloaders.imagenet_index import imagenet_index


# Load ViT Model (recommend openai/clip-vit-base-patch32 or openai/clip-vit-base-patch16)
def load_model(model_path=None, device=None):
    
    # Bringing the configs from config.py if no parameters are set!
    if model_path is None:
        model_path = config.MODEL_PATH
    if device is None:
        device = config.DEVICE
    
    model = HookedViT.from_pretrained(
        model_path, fold_ln=True, center_writing_weights=True,
        center_unembed=True, refactor_factored_attn_matrices=True )
    
    model = model.to(device)
    model.eval() # This version is eval mode (For inference)
    
    print(f"Model loaded: {model_path}")
    print(f"Device: {device}")
    
    return model

# Load the accoring SparseAutoEncoders from huggingface
# Check https://huggingface.co/Prisma-Multimodal for more models
# NOTE: If you want to use different SAEs, must make it align with the ViT you are using
#   e.g. CLIP-ViT-B32 <-> CLIP-ViT-SAE-B32 , DINO <-> DINO-SAE
def load_sae_weights(layers=None, sae_paths=None, cache_dir=None, device=None):
    
    # Yet again without parameters use base configs
    if layers is None:
        layers = [8, 9, 10, 11]
    if sae_paths is None:
        sae_paths = config.SAE_PATHS
    if cache_dir is None:
        cache_dir = config.CACHE_DIR
    if device is None:
        device = config.DEVICE
    
    sae_dict = {}
    
    for layer in layers:
        repo_id = sae_paths[layer]
        print(f"Loading Layer {layer} SAE from {repo_id}...")
        
        files = list_repo_files(repo_id)
        pt_files = [f for f in files if f.endswith('.pt')]
        
        if not pt_files:
            # Check the huggingface repository if the targeted SAE weights has the correct names
            raise FileNotFoundError(f"No .pt file found in {repo_id}")
        
        weights_file = pt_files[0]
        
        weights_path = hf_hub_download(
            repo_id=repo_id, 
            filename=weights_file,
            cache_dir=cache_dir )
        
        # config_path = hf_hub_download( repo_id=repo_id, filename="config.json", cache_dir=cache_dir )
        
        sae = SparseAutoencoder.load_from_pretrained(weights_path)
        sae = sae.to(device)
        sae.eval()
        sae_dict[layer] = sae
        
        print(f"Layer {layer} SAE loaded")
    
    print(f"\nAll SAEs loaded for layers: {layers}")
    return sae_dict

# This function is for calculating the accuracy in CLIP-based image-text similarity method.
def get_txt_emb(model_name=None, device=None, num_classes=None, class_index_dict=None):
    
    if model_name is None:
        model_name = config.MODEL_PATH
    if device is None:
        device = config.DEVICE
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if class_index_dict is None:
        class_index_dict = imagenet_index # Defaulat : ImageNet-1K
    
    class_names = []
    for i in range(num_classes):
        if str(i) in class_index_dict:
            class_name = class_index_dict[str(i)][1]
            class_names.append(class_name)
        else:
            raise ValueError(f"Class index {i} not found in class_index_dict")
    
    if model_name.startswith("open-clip:"):
        import open_clip
        repo_path = model_name.split(':', 1)[1]
        new_model_id = f'hf-hub:{repo_path}'
        clip_model, _, _ = open_clip.create_model_and_transforms(new_model_id)
        tokenizer = open_clip.get_tokenizer(new_model_id)
    else:
        from transformers import CLIPModel, CLIPTokenizer
        clip_model = CLIPModel.from_pretrained(model_name)
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
    
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    # NOTE: The usual text prompt
    prompts = [f"a photo of a {class_name}" for class_name in class_names]
    
    if hasattr(tokenizer, '__module__') and 'open_clip' in str(tokenizer.__class__):
        text_tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
    else:
        text_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)
    
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features