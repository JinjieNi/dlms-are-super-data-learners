"""
Convert megatron checkpoints to huggingface weights.

This script will also convert the tokenizer configured.
"""

import gc
import shutil
import os
import sys
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from argparse import ArgumentParser, Namespace
sys.path.append(str(Path(__file__).parent.parent.absolute()))

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

PROJECT_DIR = os.environ.get("PROJECT_DIR")

def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)

def convert_wqkv(llama_mega, layer_idx=0, n_heads=32, n_heads_kv=8):
    qkv_w = llama_mega[f'decoder.layers.{layer_idx}.self_attention.linear_qkv.weight']

    np = n_heads
    ng = n_heads_kv
    h = qkv_w.size(1)
    hn = h//np
    
    qkv_w = qkv_w.view(ng, -1, h).contiguous()
    assert qkv_w.shape == (ng, h//ng + 2*hn, h), f"expected shape: {(ng, h//ng + 2*hn, h)}, got shape: {qkv_w.shape}"
    
    wq, wk, wv = torch.split(qkv_w, [h//ng, hn, hn], dim=1)
    assert wq.shape == (ng, h//ng, h)
    assert wk.shape == (ng, hn, h)
    assert wv.shape == (ng, hn, h)

    wq = wq.contiguous().view(-1, h)
    wk = wk.contiguous().view(-1, h)
    wv = wv.contiguous().view(-1, h)
    return wq, wk, wv


def convert_wqkv_bias(llama_mega, n_hidden, layer_idx=0, n_heads=32, n_heads_kv=8):
    qkv_b = llama_mega[f'decoder.layers.{layer_idx}.self_attention.linear_qkv.bias']
    np = n_heads
    ng = n_heads_kv
    hn = n_hidden//np
    h = n_hidden
    
    qkv_b = qkv_b.view(ng, -1).contiguous()
    assert qkv_b.shape == (ng, h//ng + 2*hn)
    
    bq, bk, bv = torch.split(qkv_b, [h//ng, hn, hn], dim=1)
    assert bq.shape == (ng, h//ng)
    assert bk.shape == (ng, hn)
    assert bv.shape == (ng, hn)

    bq = bq.contiguous().view(-1)
    bk = bk.contiguous().view(-1)
    bv = bv.contiguous().view(-1)
    return bq, bk, bv


def convert_ffn(llama_mega, layer_idx=0, n_dense=11008):
    mega_ffn = llama_mega[f'decoder.layers.{layer_idx}.mlp.linear_fc1.weight']
    ffn_w3, ffn_w1 = mega_ffn.split(n_dense, dim=0)
    return ffn_w1, ffn_w3


def convert_ffn_router_experts(llama_mega, layer_idx=0, expert_idx=0, n_dense=11008):
    mega_ffn = llama_mega[f'decoder.layers.{layer_idx}.mlp.experts.linear_fc1.weight{expert_idx}']
    ffn_w3, ffn_w1 = mega_ffn.split(n_dense, dim=0)
    return ffn_w1, ffn_w3


def convert_ffn_shared_experts(llama_mega, layer_idx=0, n_dense=11008):
    mega_ffn = llama_mega[f'decoder.layers.{layer_idx}.mlp.shared_experts.linear_fc1.weight']
    ffn_w3, ffn_w1 = mega_ffn.split(n_dense, dim=0)
    return ffn_w1, ffn_w3


def convert_ffn_shared_experts_separate(s_ep_id, llama_mega, layer_idx=0, n_dense=11008):
    mega_ffn = llama_mega[f'decoder.layers.{layer_idx}.mlp.shared_experts.{s_ep_id}.linear_fc1.weight']
    ffn_w3, ffn_w1 = mega_ffn.split(n_dense, dim=0)
    return ffn_w1, ffn_w3


def write_fromscratch_model(
    args,
    model_path,
    input_base_path,
    num_output_shards: int=2,
    norm_eps: float=1e-6,
    rope_theta: float=1e6,
    vocab_size: int=None,
):
    args1 = args
    
    # Preliminaries
    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(input_base_path, 'latest_checkpointed_iteration.txt')) as f:
        iteration = f.read()
    if iteration != "release":
        iteration = f"iter_{int(iteration):07d}"
    print(f"Fetching iteration {iteration}")

    # Load weights
    base_path = Path(input_base_path)/iteration
    assert len(list(base_path.glob("mp_rank_*"))) == 1, "Unshard your model with checkpoint_util.py first!"
    loaded = torch.load(base_path/"mp_rank_00"/"model_optim_rng.pt", map_location="cpu")
    args = loaded['args']

    loaded = loaded['model']
        
    # Load arguments
    n_layers = args.num_layers
    n_heads = args.num_attention_heads
    if args.group_query_attention:
        n_heads_kv = args.num_query_groups
    else:
        n_heads_kv = n_heads
    n_dense = args.ffn_hidden_size
    n_hidden = args.hidden_size
    hidden_per_head = n_hidden // n_heads
    intermediate_size = args.ffn_hidden_size
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, hidden_per_head, 2).float() / hidden_per_head))
    seq_length = args.seq_length

    print("Number of Layers:", n_layers)
    print("Number of Attention Heads:", n_heads)
    print("Number of Attention Heads for Key/Value:", n_heads_kv)
    print("Hidden Size for Feed-Forward Network:", n_dense)
    print("Hidden Size:", n_hidden)
    print("Hidden Size per Head:", hidden_per_head)
    print("Intermediate Size:", intermediate_size)
    print("rope_theta:", rope_theta)
    print("Inverse Frequency:", inv_freq)
    print("Sequence Length:", seq_length)

    print('Megatron ckpt loaded!')
    param_count = 0
    index_dict = {"weight_map": {}}
        
    # Start conversion
    with TemporaryDirectory() as tmp_model_path:
        print(f'Weighted Converting for {n_layers} layers...')
        for layer_i in range(n_layers):
            filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
            wq_proj, wk_proj, wv_proj = convert_wqkv(llama_mega=loaded, 
                                          layer_idx=layer_i, n_heads=n_heads,
                                          n_heads_kv=n_heads_kv)
            ffn_w1, ffn_w3 = convert_ffn(llama_mega=loaded, 
                                        layer_idx=layer_i, 
                                        n_dense=n_dense)

            # wq_proj_bias, wk_proj_bias, wv_proj_bias = convert_wqkv_bias(llama_mega=loaded, 
            #                               layer_idx=layer_i, n_heads=n_heads,
            #                               n_heads_kv=n_heads_kv, n_hidden=n_hidden)

            state_dict = {
                f"model.layers.{layer_i}.self_attn.q_proj.weight": wq_proj,
                f"model.layers.{layer_i}.self_attn.k_proj.weight": wk_proj,
                f"model.layers.{layer_i}.self_attn.v_proj.weight": wv_proj,
                # f"model.layers.{layer_i}.self_attn.q_proj.bias": wq_proj_bias,
                # f"model.layers.{layer_i}.self_attn.k_proj.bias": wk_proj_bias,
                # f"model.layers.{layer_i}.self_attn.v_proj.bias": wv_proj_bias,
                f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[f"decoder.layers.{layer_i}.self_attention.linear_proj.weight"],
                f"model.layers.{layer_i}.self_attn.q_norm.weight": loaded[f"decoder.layers.{layer_i}.self_attention.q_layernorm.weight"],
                f"model.layers.{layer_i}.self_attn.k_norm.weight": loaded[f"decoder.layers.{layer_i}.self_attention.k_layernorm.weight"],
                f"model.layers.{layer_i}.mlp.gate_proj.weight": ffn_w3,
                f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"decoder.layers.{layer_i}.mlp.linear_fc2.weight"],
                f"model.layers.{layer_i}.mlp.up_proj.weight": ffn_w1,
                f"model.layers.{layer_i}.input_layernorm.weight": loaded[f"decoder.layers.{layer_i}.self_attention.linear_qkv.layer_norm_weight"],
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[f"decoder.layers.{layer_i}.mlp.linear_fc1.layer_norm_weight"],
                f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq": inv_freq
            }


            for k, v in state_dict.items():
                index_dict["weight_map"][k] = filename
                param_count += v.numel()
            torch.save(state_dict, os.path.join(tmp_model_path, filename))
            print(f'Sharded file saved to {filename}')

        filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
        state_dict = {
            "model.norm.weight": loaded['decoder.final_layernorm.weight'],
            "lm_head.weight": loaded['output_layer.weight'] if args1.untieembed else loaded["embedding.word_embeddings.weight"],
            "model.embed_tokens.weight": loaded['embedding.word_embeddings.weight']
        }
        
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch_dtype = state_dict["lm_head.weight"].dtype
        torch.save(state_dict, os.path.join(tmp_model_path, filename))
        print(f'Sharded file saved to {filename}')

        # Write configs and save
        index_dict["metadata"] = {"total_size": param_count * 2}
        write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

        model_dir_path = args1.model_dir_paths[args1.model]
        config = AutoConfig.from_pretrained(model_dir_path, trust_remote_code=True)
        # # assert configuration matches
        assert config.hidden_size == n_hidden
        assert config.intermediate_size == intermediate_size
        assert config.num_attention_heads == n_heads

        assert config.num_hidden_layers == n_layers
        assert config.rms_norm_eps == norm_eps, f"config.rms_norm_eps: {config.rms_norm_eps}, norm_eps: {norm_eps} "
        assert config.num_key_value_heads == n_heads_kv
        # Set vocab size
        config.vocab_size = args.padded_vocab_size
        config.max_position_embeddings = seq_length
        config.save_pretrained(tmp_model_path)

        # copy the model files to the tmp_model_path except the config
        if args1.model in args1.custom_models:
            for file_name in os.listdir(model_dir_path):
                file_path = os.path.join(model_dir_path, file_name)
                if os.path.isfile(file_path) and file_name != 'config.json':
                    shutil.copy2(file_path, tmp_model_path)
        else:
            print("Not custom model, skipping copying modeling and configuration files.")
            
        # Make space so we can load the model properly now.
        del state_dict
        del loaded
        gc.collect()

        if vocab_size is None:
            vocab_size = args.padded_vocab_size
        else:
            print(f"Using vocab size {vocab_size} from tokenizer and not {args.padded_vocab_size} from args.")
            # update config
            config.vocab_size = vocab_size

        print("Loading the hf checkpoint model...")
        model = AutoModelForCausalLM.from_pretrained(
            tmp_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        model.config.vocab_size = vocab_size
        # resizes the embedding layer to the correct size
        model.resize_token_embeddings(vocab_size)
        # Avoid saving this as part of the config.
        del model.config._name_or_path

    print("Saving in the Transformers format.")
    max_num_params_per_shard = param_count*2 // max(1,(num_output_shards-1))
    model.save_pretrained(model_path, max_shard_size=max_num_params_per_shard)
    
    # copy the model files to the tmp_model_path except the config
    if args1.model in args1.custom_models:
        for file_name in os.listdir(model_dir_path):
            file_path = os.path.join(model_dir_path, file_name)
            if os.path.isfile(file_path) and file_name != 'config.json':
                shutil.copy2(file_path, model_path)
    else:
        print("Not custom model, skipping copying modeling and configuration files.")



def write_tokenizer(args: Namespace):
    
    # In megatron now we only save base model tokenizers
    if args.model in {
        "gptneox_1.7b_dlm",
        "gptneox_1.7b_ar"
        }:
        hf_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(
                PROJECT_DIR, 
                f'megatron/training/tokenizer/gptneox_tokenizer'
                )
            )
    elif args.model in {
        "gpt2_1b_dlm",
        }:
        hf_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(
                PROJECT_DIR, 
                f'megatron/training/tokenizer/gpt2_tokenizer'
                )
            )
    else:
        raise NotImplementedError(f"Model {args.model} is not supported")

    print("Final HF Tokenizer configuration:")
    print(hf_tokenizer)
    hf_tokenizer.save_pretrained(args.output_dir)

def main():
    # make sure megatron is importable

    parser = ArgumentParser()
    parser.add_argument("--input_dir", help="Location of Megatron weights",
                        required=True)
    parser.add_argument("--num_output_shards", type=int, default=1)
    parser.add_argument("--model", choices={
        "gptneox_1.7b_dlm",
        "gptneox_1.7b_ar",
        "gpt2_1b_dlm",
        },
        default="gptneox_1.7b_dlm")
    parser.add_argument("--output_dir", help="Location to write HF model and tokenizer",
                        required=True)
    parser.add_argument("--cache_dir", help="Huggingface cache_dir (optional)")
    parser.add_argument("--vocab_file", type=str, help="Path to the vocab file")
    parser.add_argument("--vocab_extra_ids_list",
                        help="comma separated list of special vocab ids to add to the tokenizer")
    parser.add_argument("--override_special_tokens", nargs="*", default=[],
                        help=("One or more arguments to override special tokens. "
                              "Syntax set as `key=value`, e.g. `eos=<|im_end|>`. "
                              "Overrides available only bos, cls, eos, mask, pad, sep, unk."))
    parser.add_argument("--rope_theta", type=float, default=1e4,
                        help="Rope theta for the model")
    parser.add_argument("--untieembed", action="store_true",
                        help="Whether to untie the embeddings and output weights")

    args = parser.parse_args()
    

    
    args.model_dir_paths = {
        "gptneox_1.7b_dlm": os.path.join(PROJECT_DIR, f"tools/weights_conversion/hf_configs/gptneox_1.7b_dlm"),
        "gptneox_1.7b_ar": os.path.join(PROJECT_DIR, f"tools/weights_conversion/hf_configs/gptneox_1.7b_ar"),
        "gpt2_1b_dlm": os.path.join(PROJECT_DIR, f"tools/weights_conversion/hf_configs/gpt2_1b_dlm"),
    }
    
    args.custom_models = [
        "gptneox_1.7b_dlm",
        "gptneox_1.7b_ar",
        "gpt2_1b_dlm"
    ]
    
    write_fromscratch_model(
        args=args, 
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        num_output_shards=args.num_output_shards,
        rope_theta=args.rope_theta,
    )
    
    write_tokenizer(args)

if __name__ == "__main__":
    main()