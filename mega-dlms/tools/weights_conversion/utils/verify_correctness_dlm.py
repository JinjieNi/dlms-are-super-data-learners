import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM

import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent.absolute()))

from megatron.training import get_args
from megatron.core.num_microbatches_calculator import update_num_microbatches
from megatron.training.arguments import parse_args
from megatron.training.initialize import initialize_megatron, set_jit_fusion_options
from megatron.training.training import setup_model_only, build_train_valid_test_data_iterators
from megatron.core.enums import ModelType
from pretrain_difflm import (model_provider, get_batch, loss_func, 
                           extra_args_provider, train_valid_test_datasets_provider
                           )
from megatron.core.transformer.enums import AttnBackend

# import debugpy
# print("Waiting for debugger attach on port 5678...")
# debugpy.listen(("0.0.0.0", 5678))
# debugpy.wait_for_client()

def get_base_model(model):
    """
    Unwraps any nested .module attributes until you reach the actual
    GPTModel (or whichever base class you are after).
    """
    while hasattr(model, "module"):
        model = model.module
    return model

def slice_for_logging(tensor: torch.Tensor):
    """
    Return just one element from each dimension (0, 0, 0, ...).
    If you prefer to keep the entire hidden dimension and only
    slice the sequence dimension, modify as needed (e.g. tensor[:, :1, :]).
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.ndim == 0:
        # scalar
        return tensor.item()
    # For example, return the first element from each dimension
    return tensor[(0, ) * tensor.ndim].item()

def register_debug_hooks(model: torch.nn.Module, logfile: str):
    """
    Attach forward hooks to every submodule of `model` so that
    we can log input and output shapes (and a small slice of
    their data) to `logfile`.
    """
    # Make sure we clear the logfile (or append, if you prefer)
    with open(logfile, "w") as f:
        f.write(f"Logging activations for model: {model.__class__.__name__}\n")

    def hook_fn(module: torch.nn.Module, module_input, module_output):
        # Module name/class
        mod_str = f"{module.__class__.__name__}"
        
        # We often get tuples for input/output (especially in HF, with multiple outputs).
        # Carefully handle them:
        if not isinstance(module_input, tuple):
            module_input = (module_input,)
        if not isinstance(module_output, tuple):
            module_output = (module_output,)

        # Gather shapes and a small slice of input data
        in_shapes = [inp.shape if isinstance(inp, torch.Tensor) else str(type(inp))
                     for inp in module_input]
        in_slices = [slice_for_logging(inp) for inp in module_input if isinstance(inp, torch.Tensor)]

        # Gather shapes and a small slice of output data
        out_shapes = [out.shape if isinstance(out, torch.Tensor) else str(type(out))
                      for out in module_output]
        out_slices = [slice_for_logging(out) for out in module_output if isinstance(out, torch.Tensor)]

        # Append info to the logfile
        if mod_str != "IdentityOp":
            with open(logfile, "a") as f:
                f.write(f"MODULE: {mod_str}\n")
                f.write(f"  Input shapes:  {in_shapes}\n")
                f.write(f"  Input sample:  {in_slices}\n")
                f.write(f"  Output shapes: {out_shapes}\n")
                f.write(f"  Output sample: {out_slices}\n\n")
        

    # Register the hook on every submodule
    for name, submodule in model.named_modules():
        # Skip registering on the top-level model container or wrap it if you prefer
        if submodule is not model:
            submodule.register_forward_hook(hook_fn)


def hf_provider(args, cache_dir: Optional[Path], device: str, bf16: bool = False):
    print("Getting huggingface model...")
    extra_kwargs = {}
    if bf16:
        extra_kwargs = {"torch_dtype": torch.bfloat16}
    else:
        extra_kwargs = {"torch_dtype": "auto"}
    
    if args.attention_backend == AttnBackend.flash:
        print("Using flash attention!")
        extra_kwargs["attn_implementation"] = "flash_attention_2"
        
    if args.use_local_hf_model:
        print("Using local hf model!")
        extra_kwargs["trust_remote_code"] = True

    model = AutoModelForCausalLM.from_pretrained(cache_dir, **extra_kwargs)
    
    # Calculate and print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Architecture:\n{model}")
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Total Parameters (B): {total_params/1e9:.2f}B\n")
    
    return model.eval().requires_grad_(False).to(device)

def hf_forward(model, batch):
    device = next(param.device for param in model.parameters())
    batch = [tensor.to(device) for tensor in batch[:5]]
    tokens, labels, loss_mask, attention_mask, position_ids = batch
    output = model(input_ids=tokens, position_ids=position_ids, labels=tokens) # tokens here as the labels should be right, as hf will deal with the label shiftin
    return output["logits"], output["loss"]


def mega_provider():
    print("Getting megatron model...")
    model, _ , _ = setup_model_only(model_provider, ModelType.encoder_or_decoder)
    assert len(model) == 1, "correctness verification only supported with unsharded models"
    model = get_base_model(model[0]).eval().requires_grad_(False)
    return model


def mega_forward(model, batch):
    tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = batch
    assert torch.all(loss_mask)
    output = model(tokens, position_ids, attention_mask, labels=labels)
    if len(output) == 3:
        loss, logits, difflm_mask = output
    else:
        loss, logits = output
    
    loss_mask[:, -1] = False # in this megatron implementation, the input labels are shifted by 1, so we need to mask the last token
    loss, num_tokens, _ = loss_func(loss_mask, model, loss)
    return logits.transpose(0, 1).contiguous(), loss / num_tokens


def verify_step(activation_sample_path, our_forward, our_model, base_forward, base_model, batch):
    
    # Example debugging file names for each pass:
    megatron_debugfile = activation_sample_path + "/log_megatron.txt"
    hf_debugfile = activation_sample_path + "/log_huggingface.txt"
    
    our_logits, our_loss = our_forward(our_model, batch)
    base_logits, base_loss = base_forward(base_model, batch)
    assert our_logits.size() == base_logits.size(), \
            f"ours={our_logits.size()}, true={base_logits.size()}"
    our_logits = our_logits.cpu()
    base_logits = base_logits.cpu()
    abs_error = torch.abs(our_logits - base_logits)
    print("Max absoulute error in the logits:",
          f"max={torch.max(abs_error)}, avg={torch.mean(abs_error)}")
    assert our_loss.size() == base_loss.size()
    our_loss = our_loss.cpu()
    base_loss = base_loss.cpu()
    loss_error = torch.abs(our_loss - base_loss)
    print(f"Abs loss error: {loss_error} "
          f"Our loss: {our_loss}, theirs: {base_loss}")
    
    print("\n\nThe activation logs are in the following files:")
    print(f"Megatron: {megatron_debugfile}")
    print(f"Huggingface: {hf_debugfile}\n\n")


def is_megatron_path(path: Path | str):
    path = Path(path) if isinstance(path, str) else path
    return (path/"latest_checkpointed_iteration.txt").exists()


def main():
    # Misc initializations
    print("Starting megatron vs huggingface verification")
    args = get_args()
    set_jit_fusion_options()

    # Determine if the provided weight is a megatron checkpoint or huggingface checkpoint
    print("Loading our model!")
    if is_megatron_path(args.load):
        our_model = mega_provider()
        our_forward = mega_forward
    else:
        raise ValueError(f"The given path does not look like a megatron checkpoint: {args.load}")

    # Load baseline model
    print("Loading baseline model!")
    base_model = hf_provider(args, args.cache_dir, args.baseline_device, bf16=args.bf16)
    base_forward = hf_forward

    # Load dataset iterator
    print("Loading dataset!")
    data_iterator, _, _ = build_train_valid_test_data_iterators(
        train_valid_test_datasets_provider
    )

    os.makedirs(args.activation_sample_path, exist_ok=True)
    
    # Now we can start the verifications
    for iteration in range(0, 10):
        print(f"Iteration {iteration}...")
        update_num_microbatches(args.consumed_train_samples)
        args.curr_iteration = iteration

        verify_step(args.activation_sample_path, our_forward, our_model, base_forward, base_model,
                    get_batch(data_iterator))


def extra_extra_args(parser):
    parser = extra_args_provider(parser)
    group = parser.add_argument_group(title="huggingface")
    group.add_argument("--huggingface-cache", type=Path, default=None, dest="cache_dir", help=(
        "If falcon, optional: path to huggingface cache. "
        "If llama2, optional: either the huggingface cache path, or "
        "the raw weight directory given by meta. "
        "If llama, optional: either the path to converted huggingface weights "
        "(use convert_llama_weights_to_hf.py) or the huggingface cache dir."
    ))
    group.add_argument("--huggingface-device", default="cuda:1", dest="baseline_device",
                       help="Device to use for the baseline model")
    group.add_argument("--activation-sample-path", type=str, 
                       help="Path to the activation sample file")
    group.add_argument("--use-local-hf-model", action="store_true",
                       help="Use local hf model")
    return parser


if __name__ == "__main__":
    
    defaults = {"micro_batch_size": 1, "global_batch_size": 1, "use_checkpoint_args": True, "train_iters": 10,
                "lr": 1.0}
    if not is_megatron_path(parse_args(extra_extra_args).load):
        defaults.update({"encoder_num_layers": 1, "hidden_size": 1, 
                         "num_attention_heads": 1, "seq_length": 2048,
                         "max_position_embeddings": 2048})
    initialize_megatron(extra_extra_args, args_defaults=defaults)
    main()