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

from PIL import Image
from tqdm import tqdm

class ViT_TTA:
    def __init__(self, SAETester, classnames, percentiles=[0]):
        self.sae_tester = SAETester # This class is also from the original code authors
        self.percentiles = percentiles
        self.device = self.sae_tester.device

        print("Initializing ViT_TTA for standard CLIP: Pre-computing text features...")
        with torch.no_grad():
            prompts = [f"a photo of a {c}" for c in classnames]
            text_inputs = self.sae_tester.vit.processor(
                text=prompts, return_tensors="pt", padding=True
            ).to(self.device)

            self.text_features = self.sae_tester.vit.model.get_text_features(**text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

            self.logit_scale = self.sae_tester.vit.model.logit_scale.exp()
            self.dtype = self.sae_tester.vit.model.dtype
        print("Initialization complete.")


    def gather_neuron(self, image, threshold=0.01, use_cls=True, use_patch=True):
        
        self.sae_tester.register_image(image)
        activations = self.sae_tester.get_activation_distribution()  # [197, 49152]

        # print(f"[DEBUG] gather_neuron: shape={activations.shape}, "
        #     f"min={activations.min():.6f}, max={activations.max():.6f}, mean={activations.mean():.6f}")

        if use_cls and not use_patch:
            
            cls_acts = activations[0]  # [49152]
            top_idx = np.argmax(cls_acts)
            print(f"[DEBUG] CLS-only neuron={top_idx}, value={cls_acts[top_idx]:.6f}")
            return top_idx, activations

        elif use_patch and not use_cls:
            
            patch_acts = activations[1:]  # [196, 49152]
            max_acts = patch_acts.max(0)  # [49152]
            top_idx = np.argmax(max_acts)
            print(f"[DEBUG] Patch-only neuron={top_idx}, max_val={max_acts[top_idx]:.6f}")
            return top_idx, activations

        else:
            
            max_acts = activations.max(0)  # [49152]
            freq = (activations > threshold).sum(0)  # [49152]
            if freq.max() > 0:
                top_idx = np.argmax(freq)
                # print(f"[DEBUG] Mixed neuron={top_idx}, freq={freq[top_idx]}, max_val={max_acts[top_idx]:.6f}")
            else:
                top_idx = np.argmax(max_acts)
                # print(f"[DEBUG] Mixed neuron={top_idx}, max_val={max_acts[top_idx]:.6f}")
            return top_idx, activations

    def create_patch_mask(self, activations, neuron_idx, threshold):
        # NOTE: This function creates the mask for dynamic inference.

        neuron_acts = activations[:, neuron_idx]  # [197]
        patch_acts = neuron_acts[1:]  # Exclude [CLS], [196]

        patch_acts_2d = patch_acts.reshape(14, 14)  # [14, 14]

        threshold_value = np.percentile(patch_acts_2d, threshold) #  (N)% percentile of the input image
        mask = patch_acts_2d > threshold_value

        return mask

    def apply_mask(self, image, mask):
        # NOTE: This function applies the mask to the input

        image_array = np.array(image)
        masked_image = image_array.copy()

        patch_h = image.height // 14
        patch_w = image.width // 14

        # TODO: I was not able to make it torch-freindly yet. Maybe later
        for i in range(14):
            for j in range(14):
                if not mask[i, j]:
                    y_start, y_end = i * patch_h, (i + 1) * patch_h
                    x_start, x_end = j * patch_w, (j + 1) * patch_w
                    masked_image[y_start:y_end, x_start:x_end] = 0

        return Image.fromarray(masked_image)

    def evaluate_single(self, image, class_names):

        self.sae_tester.register_image(image)
        pixel_values = self.sae_tester.processed_image['pixel_values'].to(self.device)

        with torch.no_grad():
            
            image_features = self.sae_tester.vit.model.get_image_features(pixel_values=pixel_values.type(self.dtype))
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity_scores = (self.logit_scale * image_features @ self.text_features.T)
            predicted_class = torch.argmax(similarity_scores, dim=-1)

        return predicted_class.cpu().item()

    def inference(self, image, class_names):

        results = {}
        current_image = image

        top_neuron_idx, initial_activations = self.gather_neuron(image)
        results['top_neuron'] = top_neuron_idx

        for phase, percentile in enumerate(self.percentiles):

            self.sae_tester.register_image(current_image)
            activations = self.sae_tester.get_activation_distribution()

            patch_mask = self.create_patch_mask(activations, top_neuron_idx, percentile)

            masked_image = self.apply_mask(current_image, patch_mask)

            results[f'phase_{phase+1}_masked_image'] = masked_image
            results[f'phase_{phase+1}_mask'] = patch_mask

            current_image = masked_image

        final_prediction = self.evaluate_single(current_image, class_names)
        results['final_prediction'] = final_prediction

        return results
    
    def inference_amp(self, image, class_names, gamma=2.0, eta=1.0, neuron_mode="patch"):

        top_neuron_idx, _ = self.gather_neuron(image)
        target_feature_indices = [top_neuron_idx]

        def amplification_hook(activations):
            sae_output_tuple, sae_cache_dict = self.sae_tester.sae.run_with_cache(activations)
            sae_reconstructed, feature_acts, loss_dict = sae_output_tuple

            before = feature_acts[0, 0, target_feature_indices[0]].item()

            amplified_features = feature_acts.clone()
            amplified_features[:, :, target_feature_indices] *= gamma
            after = amplified_features[0, 0, target_feature_indices[0]].item()

            amplified_activations = (
                amplified_features @ self.sae_tester.sae.W_dec + self.sae_tester.sae.b_dec
            )
            delta = amplified_activations - sae_reconstructed
            result = activations + eta * delta

            if neuron_mode == "patch":
                print(f"[PATCH] Neuron {target_feature_indices[0]}: {before:.4f} → {after:.4f}")
            else:
                print(f"[CLS] Neuron {target_feature_indices[0]}: {before:.4f} → {after:.4f}")
            print(f"[DEBUG] delta.norm={delta.norm():.4f}, result.norm={result.norm():.4f}")

            return (result,)

        hook = Hook(
            block_layer=self.sae_tester.cfg.block_layer,
            module_name=self.sae_tester.cfg.module_name,
            hook_fn=amplification_hook,
            is_custom=not isinstance(self.sae_tester.vit.model, CLIPModel),
            return_module_output=False,
        )

        self.sae_tester.register_image(image)
        inputs = self.sae_tester.processed_image.to(self.device)

        with torch.no_grad():
            output = self.sae_tester.vit.run_with_hooks([hook], **inputs)
            image_features = output.image_embeds
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity_scores = self.logit_scale * image_features @ self.text_features.T
            predicted_class = torch.argmax(similarity_scores, dim=-1)

        return {
            "final_prediction": predicted_class.cpu().item(),
            "target_features": target_feature_indices,
            "gamma": gamma,
            "eta": eta,
            "mode": neuron_mode,
        }
