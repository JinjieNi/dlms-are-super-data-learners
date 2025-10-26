# Example script for precision verification of the ckpt conversion. 
# Run it to verify the correctness if you are modifying the model architectures.
# logits avg at the 10^-2 level, loss avg at the 10^-2 to 10^-3 level should be fine.

DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"
MODEL_ARGS="--bf16 --attention-backend flash --seq-length 128 --hidden-dropout 0.0 --attention-dropout 0.0"
COMMON_ARGS="--attention-softmax-in-fp32 --no-bias-gelu-fusion --no-bias-dropout-fusion --no-bias-swiglu-fusion --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1"
DIFFLM_ARGS="--model-running-mode test-forward --base-model vanilla --attention-mask-type no_mask"

ckpt_dir="/jinjieni/df_ckpt/cache/difflm/converted_checkpoints/dlm_training_test/ckptstep_28500"
megatron_ckpt=$ckpt_dir/torch
hf_ckpt=$ckpt_dir/hf
validation_log_dir=$ckpt_dir/validation_logs

data_local_shard_dir="$DATASETS_DIR/nemotron_cc_extracted/processed_gpt2/1b/extracted_mixture_000001"
data_cache_dir="$CKPT_DIR/cache/difflm/data/data_cache"

CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun $DISTRIBUTED_ARGS \
    tools/weights_conversion/utils/verify_correctness_dlm.py \
	--load $megatron_ckpt \
	--data-path $data_local_shard_dir \
	--data-cache-path $data_cache_dir \
	--tokenizer-type HuggingFaceTokenizer \
	--tokenizer-model megatron/training/tokenizer/gpt2_tokenizer \
    --finetune \
    --gpt-block-return-loss-and-logits \
	--use-checkpoint-args \
    $COMMON_ARGS $MODEL_ARGS $DIFFLM_ARGS \
	--huggingface-cache $hf_ckpt \
	--huggingface-device=cuda:0 \
	--activation-sample-path $validation_log_dir \
    --use-local-hf-model \
	--bf16


