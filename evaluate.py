import torch
import tqdm

def evaluate_baseline(sae_tester, dataset, classnames, num_samples=100):
    correct = 0
    total = 0

    with torch.no_grad():
        prompts = [f"a photo of a {c}" for c in classnames]
        text_inputs = sae_tester.vit.processor(
            text=prompts, return_tensors="pt", padding=True
        ).to(sae_tester.device)

        text_features = sae_tester.vit.model.get_text_features(**text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    print(f"Loaded text features with shape: {text_features.shape}")

    for i in tqdm(range(min(num_samples, len(dataset))), desc="Measuring baseline"):
        try:
            sample = dataset[i]
            image = sample.get('image', sample.get('jpg'))
            true_label = sample.get('label', sample.get('cls'))

            if image is None or true_label is None:
                continue

            sae_tester.register_image(image)
            pixel_values = sae_tester.processed_image['pixel_values'].to(sae_tester.device)

            with torch.no_grad():
                
                image_features = sae_tester.vit.model.get_image_features(pixel_values=pixel_values)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                logit_scale = sae_tester.vit.model.logit_scale.exp()
                similarity_scores = (logit_scale * image_features @ text_features.T)
                predicted = torch.argmax(similarity_scores, dim=-1).cpu().item()

            if predicted == true_label:
                correct += 1
            total += 1

        except Exception as e:
            import traceback
            print(f"Error processing sample {i}: {e}")
            traceback.print_exc()
            continue

    if total == 0:
        print("\nError: No samples were processed successfully.")
        return 0

    print(f"\nBaseline Results: {correct}/{total} = {correct/total:.3f}")
    return correct / total

def evaluate_AMP(vit_tta, dataset, classnames, gamma=2.0, eta=1.0, num_samples=100,
                 neuron_mode="mixed", threshold=0.01):
    correct = 0
    total = 0

    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            image = sample.get('image', sample.get('jpg'))
            true_label = sample.get('label', sample.get('cls'))
            if image is None or true_label is None:
                print(f"Failed to run at {i}th iteration..")
                continue

            results = vit_tta.inference_amp(
                image,
                classnames,
                gamma=gamma,
                eta=eta,
                neuron_mode=neuron_mode
            )
            predicted = results['final_prediction']

            if predicted == true_label:
                correct += 1
            total += 1

            # if i < 5 or (i + 1) % 20 == 0:
            #     print(f"[DEBUG] Sample {i}: mode={neuron_mode}, "
            #           f"pred={predicted}, "
            #           f"correct={'✓' if predicted == true_label else '✗'}")

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    if total == 0:
        print("[ERROR] No samples processed successfully.")
        return 0.0

    acc = correct / total
    print(f"[RESULT] AMP ({neuron_mode}) - {correct}/{total} = {acc:.3f}")
    return acc
