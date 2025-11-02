import torch
from vit_prisma.dataloaders.imagenet_index import imagenet_index
from . import config
from .hooks import amp_hook


def clip_predictions(image_embeddings, text_embeddings, temperature=100.0):
    
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    similarity = temperature * image_embeddings @ text_embeddings.T
    predictions = similarity.argmax(dim=-1)
    return predictions


def class_accuracy(labels, predictions):
    
    unique_classes = labels.unique()
    per_class_stats = {}
    
    for cls in unique_classes:
        cls_idx = cls.item()
        mask = (labels == cls)
        correct = (predictions[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        per_class_stats[cls_idx] = {
            'correct': correct,
            'total': total,
            'accuracy': correct / total if total > 0 else 0.0
        }
    
    return per_class_stats


def _run_inference(model, dataloader, txt_emb, hooks=None, device=None, log_freq=100, desc=""):
    
    if device is None:
        device = config.DEVICE
    
    correct = 0
    samples = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)
            preds = clip_predictions(output, txt_emb, temperature=100.0)
            
            correct += (preds == labels).sum().item()
            samples += labels.shape[0]
            
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            
            if log_freq > 0 and (batch_idx + 1) % log_freq == 0:
                acc = correct / samples * 100
                print(f"{desc}Batch {batch_idx+1}/{len(dataloader)} | Acc: {acc:.2f}%")
    
    return correct, samples, torch.cat(all_labels), torch.cat(all_preds)


def evaluate(model, sae_dict, dataloader, target_layers, config_dict, 
             txt_emb, device=None, separate_passes=False, log_freq=100):
    """   
    Args:
        model: HookedViT
        sae_dict: {layer_idx: SAE}
        dataloader: DataLoader
        target_layers: layer list
        config_dict: {'k', 'gamma', 'eta', 'spatial_selection', 'selection_config'}
        txt_emb: text embeddings
        separate_passes: if true tjem vanilla/intervened in seperate pass (memory efficient)
    
    Returns:
        results: Result dictionary
    """
    if device is None:
        device = config.DEVICE
    
    model.eval()
    for sae in sae_dict.values():
        sae.eval()
    
    if separate_passes:
        # Vanilla pass
        v_correct, v_samples, all_labels, vanilla_preds = _run_inference(
            model, dataloader, txt_emb, device=device, log_freq=log_freq, desc="Vanilla "
        )
        
        # Intervened pass
        hooks = []
        for layer_idx in target_layers:
            hook_fn = amp_hook(
                sae=sae_dict[layer_idx], k=config_dict['k'],
                gamma=config_dict['gamma'], eta=config_dict['eta'],
                spatial_selection=config_dict['spatial_selection'],
                selection_config=config_dict.get('selection_config')
            )
            handle = model.blocks[layer_idx].hook_resid_post.register_forward_hook(hook_fn)
            hooks.append(handle)
        
        print("Running intervened pass...")
        i_correct, i_samples, _, intervened_preds = _run_inference(
            model, dataloader, txt_emb, device=device, log_freq=log_freq, desc="Intervened "
        )
        
        for handle in hooks:
            handle.remove()
            
    else:
        # Combined pass
        v_correct = 0
        i_correct = 0
        samples = 0
        all_labels = []
        vanilla_preds = []
        intervened_preds = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(device)
                labels = labels.to(device)
                
                # Vanilla
                v_output = model(images)
                v_preds = clip_predictions(v_output, txt_emb, temperature=100.0)
                
                # Intervened
                hooks = []
                for layer_idx in target_layers:
                    hook_fn = amp_hook(
                        sae=sae_dict[layer_idx], k=config_dict['k'],
                        gamma=config_dict['gamma'], eta=config_dict['eta'],
                        spatial_selection=config_dict['spatial_selection'],
                        selection_config=config_dict.get('selection_config')
                    )
                    handle = model.blocks[layer_idx].hook_resid_post.register_forward_hook(hook_fn)
                    hooks.append(handle)
                
                i_output = model(images)
                i_preds = clip_predictions(i_output, txt_emb, temperature=100.0)
                
                for handle in hooks:
                    handle.remove()
                
                v_correct += (v_preds == labels).sum().item()
                i_correct += (i_preds == labels).sum().item()
                samples += labels.shape[0]
                
                all_labels.append(labels.cpu())
                vanilla_preds.append(v_preds.cpu())
                intervened_preds.append(i_preds.cpu())
                
                if log_freq > 0 and (batch_idx + 1) % log_freq == 0:
                    v_acc = v_correct / samples * 100
                    i_acc = i_correct / samples * 100
                    print(f"Batch {batch_idx+1}/{len(dataloader)} | V: {v_acc:.2f}% | I: {i_acc:.2f}%")
        
        all_labels = torch.cat(all_labels)
        vanilla_preds = torch.cat(vanilla_preds)
        intervened_preds = torch.cat(intervened_preds)
        v_samples = i_samples = samples
    
    vanilla_accuracy = v_correct / v_samples
    intervened_accuracy = i_correct / i_samples
    accuracy_gain = intervened_accuracy - vanilla_accuracy
    
    per_class_vanilla = class_accuracy(all_labels, vanilla_preds)
    per_class_intervened = class_accuracy(all_labels, intervened_preds)
    
    return {
        'vanilla_accuracy': vanilla_accuracy,
        'intervened_accuracy': intervened_accuracy,
        'accuracy_gain': accuracy_gain,
        'total_samples': v_samples,
        'per_class_vanilla': per_class_vanilla,
        'per_class_intervened': per_class_intervened
    }