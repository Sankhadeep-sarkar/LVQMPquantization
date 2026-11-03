"""
export_weights.py
-----------------
Load a saved checkpoint and export weights to header without retraining.
Handles LVQ codebook size mismatches automatically.

Usage:
    python export_weights.py --checkpoint saved_models/resnet20_epoch2_acc0.8456_mixed_precision_lvq.pt
"""

import argparse
import torch
import numpy as np
import os

from resnet20 import resnet20
from mobilenetv2 import mobilenetv2
from inceptionv3 import inceptionv3

MODEL_REGISTRY = {
    "resnet20": resnet20,
    "mobilenetv2": mobilenetv2,
    "inceptionv3": inceptionv3,
}


def load_and_fix_checkpoint(model, checkpoint_path):
    """Load checkpoint and resize LVQ codebooks to match."""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Extract bitwidths from the checkpoint's layer names
    bitwidths = {}
    for key in state_dict.keys():
        if 'weight_bits' in key:
            parts = key.split('.')
            layer_name = '.'.join(parts[:-1])
            bitwidths[layer_name] = int(state_dict[key].item())
    
    # Resize LVQ codebooks in model to match checkpoint
    print("Resizing LVQ codebooks to match checkpoint...")
    for name, module in model.named_modules():
        if hasattr(module, 'lvq_w'):
            if name in bitwidths:
                bits = bitwidths[name]
                num_levels = max(2 ** min(bits, 16), 2)
                module.lvq_w.set_num_levels(num_levels)
                print(f"  {name}: {bits} bits -> {num_levels} levels")
            else:
                codebook_key = f"{name}.lvq_w.codebook"
                if codebook_key in state_dict:
                    num_levels = state_dict[codebook_key].shape[0]
                    bits = int(np.log2(num_levels))
                    module.lvq_w.set_num_levels(num_levels)
                    print(f"  {name}: inferred {bits} bits -> {num_levels} levels")
    
    # Now load the state dict
    print("Loading weights...")
    model.load_state_dict(state_dict, strict=False)
    
    # Get accuracy if available
    accuracy = checkpoint.get('accuracy', None)
    epoch = checkpoint.get('epoch', None)
    
    return model, accuracy, epoch


def export_weights_to_header(model, model_name, output_path):
    """Export quantized weights to C header."""
    output_path = output_path or "weights.h"
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    base, ext = os.path.splitext(output_path)
    candidate = output_path
    counter = 1
    while os.path.exists(candidate):
        candidate = f"{base}_{counter}{ext}"
        counter += 1

    header = f"""/*
 * {model_name}_weights.h
 * Extracted from trained model checkpoint.
 */

#ifndef {model_name.upper()}_WEIGHTS_H
#define {model_name.upper()}_WEIGHTS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {{
#endif

"""
    
    # Extract and write each layer's weights
    layer_count = 0
    total_weights = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight_bits') and hasattr(module, 'weight'):
            # Get quantized weights
            if hasattr(module, '_quantized_weight'):
                with torch.no_grad():
                    w = module._quantized_weight()
            else:
                w = module.weight.data
            
            # FIX: detach() before converting to numpy
            w_np = w.detach().cpu().numpy().astype(np.float32)
            flat = w_np.flatten()
            numel = len(flat)
            bits = module.weight_bits if hasattr(module, 'weight_bits') else 32
            
            var_name = f"layer_{layer_count}_{name.replace('.', '_')}"
            
            header += f"/* Layer: {name}\n"
            header += f"   Shape: {list(w_np.shape)}\n"
            header += f"   Bitwidth: {bits}\n"
            header += f"   Elements: {numel} */\n"
            header += f"static const float {var_name}[{numel}] = {{\n    "
            
            # Write weights (first 10 to keep it readable)
            if numel <= 20:
                header += ', '.join([f'{x:.8f}f' for x in flat])
            else:
                header += ', '.join([f'{x:.8f}f' for x in flat[:10]])
                header += f',\n    // ... {numel - 10} more values ...\n    '
                header += ', '.join([f'{x:.8f}f' for x in flat[-5:]])
            
            header += "\n};\n\n"
            
            layer_count += 1
            total_weights += numel
    
    header += f"#define {model_name.upper()}_NUM_LAYERS {layer_count}\n"
    header += f"#define {model_name.upper()}_TOTAL_WEIGHTS {total_weights}\n"
    
    header += """
#ifdef __cplusplus
}
#endif

#endif /* WEIGHTS_H */
"""
    
    with open(candidate, 'w') as f:
        f.write(header)
    
    print(f"\nExported {layer_count} layers, {total_weights:,} weights to {candidate}")
    print(f"Size: {total_weights * 4 / (1024 * 1024):.2f} MB (float32)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved model checkpoint (.pt file)")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="resnet20",
                        help="Model architecture (default: resnet20)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output header file path (default: weights.h)")
    args = parser.parse_args()
    
    # Create model
    print(f"Creating {args.model} model...")
    model = MODEL_REGISTRY[args.model]()
    
    # Load and fix checkpoint
    model, accuracy, epoch = load_and_fix_checkpoint(model, args.checkpoint)
    
    if accuracy is not None:
        print(f"Checkpoint accuracy: {accuracy:.4f}")
    if epoch is not None:
        print(f"Checkpoint epoch: {epoch}")
    
    # Export
    output_path = args.output or "weights.h"
    export_weights_to_header(model, args.model, output_path)


if __name__ == "__main__":
    main()