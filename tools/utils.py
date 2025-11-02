import random
import torch
import numpy as np
import pickle
import json

from datetime import datetime
from vit_prisma.dataloaders.imagenet_index import imagenet_index


def quick_stratified_split(class_indices, val_size=10000, seed=42):
    
    torch.manual_seed(seed)
    random.seed(seed)
    
    num_classes = len(class_indices)
    samples_per_class = val_size // num_classes
    val_indices = []
    
    for label, indices in class_indices.items():
        if len(indices) >= samples_per_class:
            selected = random.sample(indices, samples_per_class)
        else:
            selected = indices
        val_indices.extend(selected)
    
    return val_indices


def save_results(results, output_path, format='pickle'):
   
    if format == 'pickle':
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
    elif format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"Results saved to: {output_path}")


def load_results(input_path, format='pickle'):
    
    if format == 'pickle':
        with open(input_path, 'rb') as f:
            results = pickle.load(f)
    elif format == 'json':
        with open(input_path, 'r') as f:
            results = json.load(f)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return results


def print_results(results, top_n=10):
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Vanilla Accuracy: {results['vanilla_accuracy']*100:.2f}%")
    print(f"Intervened Accuracy: {results['intervened_accuracy']*100:.2f}%")
    print(f"Accuracy Gain: {results['accuracy_gain']*100:+.2f}%")
    print(f"Total Samples: {results['total_samples']}")
    print("="*60)
    
    if 'per_class_vanilla' in results and 'per_class_intervened' in results:
        vanilla_acc = results['per_class_vanilla']
        intervened_acc = results['per_class_intervened']
        
        gains = {}
        for cls_idx in vanilla_acc.keys():
            v_acc = vanilla_acc[cls_idx]['accuracy']
            i_acc = intervened_acc[cls_idx]['accuracy']
            gains[cls_idx] = i_acc - v_acc
        
        sorted_gains = sorted(gains.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        print(f"\nTop {top_n} Most Improved Classes:")
        for cls_idx, gain in sorted_gains:
            cls_name = imagenet_index[str(cls_idx)][1]
            v_acc = vanilla_acc[cls_idx]['accuracy'] * 100
            i_acc = intervened_acc[cls_idx]['accuracy'] * 100
            print(f"{cls_name:30s}: {v_acc:5.1f}% -> {i_acc:5.1f}% ({gain*100:+.1f}%)")


def set_seed(seed=42):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False