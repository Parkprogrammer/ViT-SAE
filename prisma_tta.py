import argparse
import os
import sys
import datetime

from tools import config, models, data, evaluation, utils


def parse_args():
    parser = argparse.ArgumentParser(description='Test-Time Adaptation using SAE')
    
    # Dataset Configuration
    parser.add_argument('--data_path', type=str, default=config.DATA_PATH,
                       help='Dataset Path')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                       help='Batch Size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader worker')
    parser.add_argument('--subset_size', type=int, default=None,
                       help='Number of samples to use')
    
    # Model Configuration
    parser.add_argument('--model_path', type=str, default=config.MODEL_PATH,
                       help='Model path')
    parser.add_argument('--layers', type=int, nargs='+', default=[9, 10, 11],
                       help='Layers to implement')
    parser.add_argument('--cache_dir', type=str, default=config.CACHE_DIR,
                       help='HuggingFace cache directory')
    
    # Intervention Configuration
    parser.add_argument('--k', type=int, default=1,
                       help='Top-K features')
    parser.add_argument('--gamma', type=float, default=1.5,
                       help='Amplification coefficient')
    parser.add_argument('--eta', type=float, default=1.0,
                       help='Delta scaling coefficient')
    parser.add_argument('--no_spatial_selection', action='store_true',
                       help='Just true always')
    parser.add_argument('--selection_method', type=str, default='topk',
                       choices=['topk', 'threshold', 'adaptive'],
                       help='Patch selection policy(Recommend topk)')
    parser.add_argument('--top_k_percent', type=float, default=0.4,
                       help='Top-k ratio')
    
    # Compuation Config
    parser.add_argument('--separate_passes', action='store_true',
                       help='Efficient Evaluation')
    parser.add_argument('--log_freq', type=int, default=100,
                       help='Logging for interation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed num')
    parser.add_argument('--device', type=str, default=config.DEVICE,
                       help='Device')
    
    # Output Config
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Result save directory path')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results or not')
    
    return parser.parse_args()


def main():
    
    args = parse_args()
    utils.set_seed(args.seed)
    
    print("Test-Time Adaptation using SAE")
    
    print("\n1. Loading model...")
    model = models.load_model(args.model_path, args.device)
    
    print("\n2. Loading SAE weights...")
    sae_dict = models.load_sae_weights(
        layers=args.layers,
        cache_dir=args.cache_dir,
        device=args.device
    )
    
    print("\n3. Generating text embeddings...")
    txt_emb = models.get_txt_emb(args.model_path, args.device)
    print(f"Text embeddings shape: {txt_emb.shape}")
    
    print("\n4. Loading dataset...")
    dataset = data.load_dataset(
        data_path=args.data_path,
        subset_size=args.subset_size
    )
    
    dataloader = data.create_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    eval_config = {
        'k': args.k,
        'gamma': args.gamma,
        'eta': args.eta,
        'spatial_selection': not args.no_spatial_selection,
        'selection_config': {
            'method': args.selection_method,
            'top_k_percent': args.top_k_percent
        }
    }
    
    print("\n5. Evaluation config:")
    print(f"   Layers: {args.layers}")
    print(f"   k: {args.k}, gamma: {args.gamma}, eta: {args.eta}")
    print(f"   Spatial selection: {eval_config['spatial_selection']}")
    print(f"   Selection method: {args.selection_method} (top_k: {args.top_k_percent})")
    
    print("\n6. Running evaluation...")
    results = evaluation.evaluate(
        model=model,
        sae_dict=sae_dict,
        dataloader=dataloader,
        target_layers=args.layers,
        config_dict=eval_config,
        txt_emb=txt_emb,
        device=args.device,
        separate_passes=args.separate_passes,
        log_freq=args.log_freq
    )
    
    utils.print_results(results)
    
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(args.output_dir, f'results_{timestamp}.pkl')
        
        results['config'] = eval_config
        results['args'] = vars(args)
        
        utils.save_results(results, output_path, format='pickle')
        
        json_results = {
            'config': eval_config,
            'vanilla_accuracy': results['vanilla_accuracy'],
            'intervened_accuracy': results['intervened_accuracy'],
            'accuracy_gain': results['accuracy_gain'],
            'total_samples': results['total_samples']
        }
        json_path = os.path.join(args.output_dir, f'results_{timestamp}.json')
        utils.save_results(json_results, json_path, format='json')
    
    print("\nDone!")


if __name__ == '__main__':
    main()