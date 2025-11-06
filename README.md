<p align="center" width="100%">
<img src="resources/imgs/3.png"  width="100%" height="100%">
</p>

<div align="center">

**Diffusion Language Models are Super Data Learners**
===========================

[Jinjie Ni†](https://jinjieni.github.io/), [Qian Liu](https://scholar.google.com/citations?user=bcbeUo0AAAAJ&hl=en), [Longxu Dou](https://longxudou.github.io/), [Chao Du](https://duchao0726.github.io/), [Zili Wang](https://commencement.github.io/), [Hang Yan](https://scholar.google.com/citations?user=yigHzW8AAAAJ&hl=en), [Tianyu Pang](https://p2333.github.io/), [Michael Qizhe Shieh](https://michaelshieh.com/)

†Correspondence to: Jinjie Ni \<jinjieni@nus.edu.sg\>

---

<h4>Witness the first crossover where diffusion beats AR.</h4>

[![Static Badge](https://img.shields.io/badge/Paper-arXiv-darkred)](https://arxiv.org/abs/2511.03276)
[![Static Badge](https://img.shields.io/badge/Blog-2025--08--09-darkcyan)](https://jinjieni.notion.site/Diffusion-Language-Models-are-Super-Data-Learners-239d8f03a866800ab196e49928c019ac)
[![Static Badge](https://img.shields.io/badge/Infra-training--backend-black)](https://github.com/JinjieNi/MegaDLMs)
[![Static Badge](https://img.shields.io/badge/Resources-ckpts--logs-green)](#resources)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=tweet1)](https://x.com/NiJinjie/status/1954177095435014533)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=tweet2)](https://x.com/NiJinjie/status/1974113126905000242)

</div>



# News
[2025-10-27] We release the codebase, all training checkpoints, and logs. The codebase is highly optimized and is industry-level in terms scalability and efficiency.

[2025-10-03] The full paper is out! Check it out [here](resources/pdf/Diffusion_Language_Models_are_Super_Data_Learners.pdf)! We did extensive ablations and scaled-up runs.


<br>

# Code
The codebase is released [here](https://github.com/JinjieNi/MegaDLMs). It is a highly-optimized codebase for any-scale DLMs training backend with Megatron-LM.

> You can also use the code under the `mega-dlms` folder of this repo, which might not be actively maintained.

<br>

# Resources

We opensource all model checkpoints and training logs mentioned in the paper. All of them can be downloaded at https://huggingface.co/collections/jinjieni/mdga.

The easiest way to download a folder is using this script (setup the variables properly):
```
python utils/hf_download_folder.py
```

Alternatively, you can also use `wget` to directly download individual files from the folder, e.g.:
```bash
wget https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/xxx/xxx.pt
```

We link the related resources below:

- Diffusion vs. AR with various unique data budgets
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/0b5_192e_dlm)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_data_budget/0b5_192e_difflm_1b)] 0.5B unique 192 epochs DLM
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/0b5_192e_ar)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_data_budget/0b5_192e_ar_1b)] 0.5B unique 192 epochs AR
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_dlm)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_data_budget/1b_96e_difflm_1b)] 1B unique 96 epochs DLM
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_ar)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_data_budget/1b_96e_ar_1b)] 1B unique 96 epochs AR
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b5_64e_dlm)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_data_budget/1b5_64e_difflm_1b)] 1.5B unique 64 epochs DLM
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b5_64e_ar)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_data_budget/1b5_64e_ar_1b)] 1.5B unique 64 epochs AR
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/10b_9e6_dlm)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_data_budget/10b_9e6_difflm_1b)] 10B unique 9.6 epochs DLM
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/10b_9e6_ar)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_data_budget/10b_9e6_ar_1b)] 10B unique 9.6 epochs AR
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/96b_1e_dlm)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_data_budget/96b_1e_difflm_1b)] 96B unique 1 epochs DLM 
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/96b_1e_ar)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_data_budget/96b_1e_ar_1b)] 96B unique 1 epochs AR
- Diffusion vs. AR with various data qualities
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_dlm_1b_low)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_data_quality/1b_96e_difflm_1b_low)] low DLM
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_ar_1b_low)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_data_quality/1b_96e_ar_1b_low)] low AR
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_dlm_1b_medium)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_data_quality/1b_96e_difflm_1b_medium)] medium DLM
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_ar_1b_medium)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_data_quality/1b_96e_ar_1b_medium)] medium AR
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_dlm)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_data_quality/1b_96e_difflm_1b_high)] high DLM
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_ar)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_data_quality/1b_96e_ar_1b_high)] high AR
- Diffusion vs. AR with various model sizes
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_dlm)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_model_size/1b_96e_difflm_1b)] 1B DLM
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_ar)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_model_size/1b_96e_ar_1b)] 1B AR
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_dlm_2b_1)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_model_size/1b_96e_difflm_2b)] 2B DLM
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_ar_2b)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_model_size/1b_96e_ar_2b)] 2B AR
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_dlm_4b)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_model_size/1b_96e_difflm_4b)] 4B DLM
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_ar_4b)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_model_size/1b_96e_ar_4b)] 4B AR
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_dlm_8b)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_model_size/1b_96e_difflm_8b)] 8B DLM
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_ar_8b)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_model_size/1b_96e_ar_8b)] 8B AR
- Diffusion vs. AR with various model sparsities
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_dlm)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_sparsity/1b_96e_difflm_1b1a)] 1B DLM
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_ar)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_sparsity/1b_96e_ar_1b1a)] 1B AR
    - [[ckpt](https://huggingface.co/datasets/MDGA-2/openmoe2_ckpts/tree/main/arch_exps/8b1a_1b96e_tc_dlm)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_sparsity/1b_96e_difflm_8b1a)] 8B1A DLM
    - [[ckpt](https://huggingface.co/datasets/MDGA-2/openmoe2_ckpts/tree/main/arch_exps/8b1a_1b96e_tc_ar)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_sparsity/1b_96e_ar_8b1a)] 8B1A AR
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_dlm_8b)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_sparsity/1b_96e_difflm_8b8a)] 8B DLM
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_ar_8b)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/vary_sparsity/1b_96e_ar_8b8a)] 8B AR
- Input Masking
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_ar)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/mask_input/1b_96e_ar_0.0)] 0.0
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_ar_1b_mask_0.1)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/mask_input/1b_96e_ar_0.1)] 0.1
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_ar_1b_mask0.3)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/mask_input/1b_96e_ar_0.3)] 0.3
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_ar_1b_mask0.5)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/mask_input/1b_96e_ar_0.5)] 0.5
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_ar_1b_mask0.7)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/mask_input/1b_96e_ar_0.7)] 0.7
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_ar_1b_mask0.9)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/mask_input/1b_96e_ar_0.9)] 0.9
- Dropout Ablations
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_96e_ar)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/dropout/0b5_192e_ar_0.0)] 0.0
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/ar_dropout_0.1)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/dropout/0b5_192e_ar_0.1)] 0.1
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/ar_dropout_0.2)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/dropout/0b5_192e_ar_0.2)] 0.2
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/ar_dropout_0.3)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/dropout/0b5_192e_ar_0.3)] 0.3
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/ar_dropout_0.4)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/dropout/0b5_192e_ar_0.4)] 0.4
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/ar_dropout_0.5)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/dropout/0b5_192e_ar_0.5)] 0.5
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/ar_dropout_0.7)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/dropout/0b5_192e_ar_0.7)] 0.7
    - [[ckpt](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/ar_dropout_0.9)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/dropout/0b5_192e_ar_0.9)] 0.9
- Coder Scaling
    - [[constant ckpts (HF)](https://huggingface.co/datasets/jinjieni/l_ckpt/tree/main/difflm/converted_checkpoints/coder_scaling_ar_2b_stage1_1)][[anneal ckpts (HF)](https://huggingface.co/datasets/jinjieni/l_ckpt/tree/main/difflm/converted_checkpoints/coder_scaling_ar_2b_stage2_from_iter_319456_1)][[constant log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/coder_scaling/10b_144_dlm_constant)][[anneal log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/coder_scaling/10b_10_dlm_anneal)] 1.7B DLM
    - [[constant ckpts (HF)](https://huggingface.co/datasets/jinjieni/l_ckpt/tree/main/difflm/converted_checkpoints/coder_scaling_dlm_2b_stage1_1)][[anneal ckpts (HF)](https://huggingface.co/datasets/jinjieni/l_ckpt/tree/main/difflm/converted_checkpoints/coder_scaling_dlm_2b_stage2_from_iter_319456_1)][[constant log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/coder_scaling/10b_144_ar_constant)][[anneal log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/coder_scaling/10b_10_ar_anneal)] 1.7B AR
- [[ckpt (HF)](https://huggingface.co/datasets/MDGA-1/super_data_learners_ckpts/tree/main/1b_480e_dlm)][[log](https://huggingface.co/datasets/MDGA-1/super_data_learners_logs/tree/main/scaled_runs/1b_480e_difflm_1b)] 1B DLM 480 epochs

You can refer to [this](https://github.com/JinjieNi/MegaDLMs/blob/main/examples/dlm_generation/dlm_inference.py) script to inference with the huggingface checkpoints. Due to the large amount, most small checkpoints above are still in megatron formats. You may refer to [this](https://github.com/JinjieNi/MegaDLMs/blob/main/examples/dlm_training/ckpt_conversion.sh) script to convert them (need to tweak the conversion scripts).

<br>

# Citation
```
@article{ni2025superdatalearner,
  title={Diffusion Language Models are Super Data Learners},
  author={Ni, Jinjie and Liu, Qian and Dou, Longxu and Du, Chao and Wang, Zili and Yan, Hang and Pang, Tianyu and Shieh, Michael Qizhe},
  journal={arXiv preprint arXiv:2511.03276},
  year={2025}
}
```
