import torch
from collections import defaultdict

def select_active_patches(feature_acts, method='topk', threshold=None, 
                            top_k=0.3, include_cls=False):
    
    """
    Args:
        feature_acts: [batch, num_tokens, num_features]
        method: 'topk' | 'threshold' | 'adaptive'
        threshold: threshold (if method='threshold')
        top_k: top-k ratio (if method='topk')
        include_cls: Include [CLS] or not 
    
    Returns:
        mask: [batch, num_tokens]
        num_active: Number of activated pathces
    """

    batch_size, num_tokens, num_features = feature_acts.shape
    patch_activation = feature_acts.abs().sum(dim=-1)  # [batch, num_tokens]
    
    # Remove [CLS] token
    cls_strength = patch_activation[:, 0:1]  # [batch, 1]
    selected_patches = patch_activation[:, 1:]

    if method == 'topk':
        patchNum = selected_patches.shape[1]
        k = max(1, int(patchNum * top_k))
        topk_val, topk_indx = torch.topk(selected_patches, k, dim=-1)
        spatial_mask = torch.zeros_like(selected_patches, dtype=torch.bool)
        spatial_mask.scatter_(1, topk_indx, True)
        
    elif method == 'threshold':
        if threshold is None:
            raise ValueError("threshold must be provided when method='threshold'")
        spatial_mask = selected_patches > threshold
        
    elif method == 'adaptive':
        mean_strength = selected_patches.mean(dim=1, keepdim=True)
        std_strength = selected_patches.std(dim=1, keepdim=True)
        adaptive_threshold = mean_strength + 0.5 * std_strength
        spatial_mask = selected_patches > adaptive_threshold
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if include_cls:
        cls_mask = torch.ones_like(cls_strength, dtype=torch.bool)
    else:
        cls_mask = torch.zeros_like(cls_strength, dtype=torch.bool)
    
    full_mask = torch.cat([cls_mask, spatial_mask], dim=1)
    num_active = full_mask.sum().item()
    
    return full_mask, num_active

def amp_hook(sae, k, gamma, eta, spatial_selection=True, 
                selection_config=None):
    
    """
    Feature amplification hook
    
    Args:
        sae: SparseAutoencoder
        k: Top-K features
        gamma: Amplification coefficient
        eta: Delta scaling coefficient
        spatial_selection: 
        selection_config: select_active_patches
    
    Returns:
        hook_fn: Forward hook function
    """
    
    def hook_fn(module, input, output):
        cfg = selection_config if selection_config is not None else {
            'method': 'topk', 
            'top_k': 0.3
        }
        
        activations = output
        _, feature_acts, _, _, _, _, _ = sae.forward(activations)
        
        # Select Top-K features
        spatial_features = feature_acts[:, 1:, :]
        mean_features = spatial_features.mean(dim=1)
        top_values, top_indices = torch.topk(mean_features[0], k)
        target_feature_indices = top_indices
        
        # Feature amplification
        # NOTE: The most important algorithm in this directory
        amplified_features = feature_acts.clone()
        
        if spatial_selection:
            active_mask, num_active = select_active_patches(
                feature_acts, method=cfg['method'],
                top_k=cfg.get('top_k_percent', cfg.get('top_k', 0.3)),
                include_cls=False )
            
            for token_idx in range(amplified_features.shape[1]):
                if active_mask[0, token_idx]:
                    amplified_features[:, token_idx, target_feature_indices] *= gamma
        else:
            amplified_features[:, :, target_feature_indices] *= gamma
        
    
        # SAE reconstruction
        sae_recon = sae.decode(feature_acts)
        amp_recon = sae.decode(amplified_features)
        
        # Delta
        delta = amp_recon - sae_recon
        result = activations + eta * delta
        
        return result
    
    return hook_fn
        
