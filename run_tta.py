import torch
import torch.nn.functional as F
import numpy as np

from transformers import CLIPModel
from src.demo.core import SAETester
from tasks.utils import (
    get_all_classnames,
    get_max_acts_and_images,
    get_sae_and_vit,
    load_datasets,
)
from src.sae_training.hooked_vit import Hook

from vit_tta import ViT_TTA
from evaluate import evaluate_AMP, evaluate_baseline

from PIL import Image
from tqdm import tqdm


if __name__ == "__main__":

    datasets = load_datasets(include_imagenet=True)
    classnames = get_all_classnames(datasets, data_root="configs/classnames")

    root = "out/feature_data"
    sae_runname = "sae_base"
    vit_name = "base"
    sae_path = "data/sae_weight/base/out.pt"

    sae, vit, cfg = get_sae_and_vit(
        sae_path=sae_path,
        vit_type='base',
        device='cuda',
        backbone='openai/clip-vit-base-patch16'
    )
    sae = sae.to('cuda')
    
    max_act_imgs, mean_acts = get_max_acts_and_images(datasets, root, sae_runname, vit_name)
    sae_tester = SAETester(vit, cfg, sae, mean_acts, max_act_imgs, datasets, classnames)
    sae_tester.device = 'cuda'
    vit_tta = ViT_TTA(sae_tester, classnames['imagenet'], percentiles=[0])
    
    print("Measuring baseline accuracy...")

    baseline_acc_imagenet = evaluate_baseline(sae_tester, datasets['imagenet'], classnames['imagenet'], num_samples=500)
    baseline_acc_sketch = evaluate_baseline(sae_tester, datasets['imagenet-sketch'], classnames['imagenet-sketch'], num_samples=500)
    
    amp_acc_imagenet = evaluate_AMP(
        vit_tta,
        datasets['imagenet'],
        classnames['imagenet'],
        gamma=2.0,
        eta=1.0,
        num_samples=500,
        neuron_mode="patch"
    )
    print(f"Improvement : {amp_acc_imagenet - baseline_acc_imagenet:.3f}")
    
    amp_acc_sketch = evaluate_AMP(
        vit_tta,
        datasets['imagenet-sketch'],
        classnames['imagenet-sketch'],
        gamma=2.0,
        eta=1.0,
        num_samples=500,
        neuron_mode="patch"
    )
    print(f"Improvement : {amp_acc_sketch - baseline_acc_sketch:.3f}")
    
    