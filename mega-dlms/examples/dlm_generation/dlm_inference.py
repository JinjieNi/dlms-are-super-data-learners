from transformers import AutoTokenizer, AutoModelForCausalLM

# todo: point the model_dir_path to the dir containing your converted hf checkpoint.
model_dir_path = '/jinjieni/df_ckpt/cache/difflm/converted_checkpoints/dlm_training_test/ckptstep_28500/hf'
prompts = ["The capital of France is"]

tokenizer = AutoTokenizer.from_pretrained(model_dir_path)
model = AutoModelForCausalLM.from_pretrained(
            model_dir_path,
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        ).to('cuda')

model_inputs = tokenizer(prompts, return_tensors="pt", padding_side='left', padding=True).to('cuda')

generated_ids = model.generate(
    **model_inputs,
    temperature = 0.0,
    cfg = 0.0,
    remasking = "low_confidence",
    inference_block_size = 256,
    sample_steps = 256,
    max_new_tokens = 256,
    verbose=True
    )

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(response)