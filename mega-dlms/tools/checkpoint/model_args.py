

def extra_args_provider_plmt(parser):
    """plmt arguments."""
    # common
    group = parser.add_argument_group(title='plmt')
    group.add_argument('--debug', action='store_true',
                       help='Whether to run in debug mode.')
    group.add_argument('--turn-on-debugpy', action='store_true',
                       help='Whether to turn on debugpy.')
    group.add_argument('--detect-anomaly', action='store_true',
                       help='Whether to detect anomaly.')
    group.add_argument('--model-running-mode', type=str, default="plmt-lnwp", choices=["plmt", "plmt-lnwp", "vanilla"],
                       help='Running mode, such as plmt, sft, rl, or inference.')
    group.add_argument('--base-model', type=str, default="Qwen_2", choices=["vanilla", "Qwen_2"],
                       help='Use Qwen 2 series model by default (including Qwen 2.5, Qwen 2.5 Math, etc.). This only specifies the model series, instead of the specific model name.')
    group.add_argument('--n-steps-per-stage', type=int, default=1,
                       help='The number of steps per curriculum stage. 1 means the number of curriculum stages equal to the total training steps.')
    group.add_argument('--curriculum-iters-ratio', type=float, default=1.0,
                       help='The ratio of the number of steps per curriculum stage to the total training steps. 1.0 means the number of curriculum stages equal to the total training steps.')
    group.add_argument('--use-latent-validation', action='store_true', 
                       help='Whether to use the latent span corruption for validation.')
    group.add_argument('--num-latent-validation-splits', type=int, default=5,
                       help='Number of fixed corruption splits to use during validation')
    group.add_argument('--embedding-hidden-fusion-type', type=str, default="none", choices=["none", "add", "weight_fusion"],
                       help='The method to fuse the embedding and last-step output hidden states. Default is none, i.e., only using the output hidden state as the input of the next position.')
    group.add_argument('--use-embedding-hidden-weight-fusion-bias', action='store_true',
                       help='Whether to add bias to the embedding-hidden fusion layer.')
    
    # plmt
    group.add_argument('--min-p', type=float, default=0.1,
                       help='The minimum expected percentage of the latent span corruption.')
    group.add_argument('--min-l', type=int, default=2,
                       help='The minimum expected latent span corruption length. In the current implementation, an n-length latent span will perform n-1 intermeditate continuous cot. So starting from 2 means the starting length of contious cot is 1.')
    group.add_argument('--max-p', type=float, default=0.5,
                       help='The maximum percentage of the latent span corruption.')
    group.add_argument('--max-l', type=int, default=10,
                       help='The maximum latent span corruption length.')
    group.add_argument('--sigma-p', type=float, default=0.0,
                       help='The standard deviation for the percentage of the latent span corruption sampling from the shifted normal distribution. A larger value means each sampling is more deterministic.')
    group.add_argument('--sigma-l', type=float, default=0.0,
                       help='The standard deviation for the latent span corruption length sampling from the shifted normal distribution. A larger value means each sampling is more deterministic.')
    group.add_argument('--p-curriculum', type=str, default="cosine", choices=["cosine"],
                       help='The curriculum scheduling model for the percentage of the latent span corruption. Default is cosine, i.e., put more steps to the initial and target steps.')
    group.add_argument('--l-curriculum', type=str, default="cosine", choices=["cosine"],
                       help='The curriculum scheduling model for the latent span corruption length. Default is cosine, i.e., put more steps to the initial and target steps.')
    group.add_argument('--no-mask-latent-span-loss', action='store_false', dest='mask_latent_span_loss',
                       help='Whether to mask the loss for the latent span output.')
    
    # latent-next-word-prediction
    group.add_argument('--max-shift', type=int, default=10, 
                       help='The maximum number of steps to shift the target left to do the latent next word prediction (latent depth).')
    group.add_argument('--min-shift', type=int, default=2, 
                       help='The minimum number of steps to shift the target left to do the latent next word prediction (latent depth).')
    group.add_argument('--shift-curriculum', type=str, default="cosine", choices=["cosine"],
                       help='The curriculum scheduling model for the number of steps to shift the target left. Default is cosine, i.e., put more steps to the initial and target steps.')
    group.add_argument('--no-prefix-tokens-without-grad', action='store_false', dest='prefix_tokens_without_grad',
                       help='If set, it will compute loss and grad for the prefix tokens (the normal prefix span due to the shifted left of the target).')
    group.add_argument('--no-mixed-depth-supervision', action='store_false', dest='mixed_depth_supervision',
                       help='If set, it will not aggregate the teacher forcing loss for each depth of latent output. Setting to False means only computing the last span loss.')
    group.add_argument('--n-truncated-backprop', type=int, default=2,
                       help='The depths to backpropagate among along the latent depths. We count from the last depth by default.')
    group.add_argument('--add-vanilla-ntp-validation', action='store_true', 
                       help='Whether to add the vanilla NTP validation.')
    group.add_argument('--normal-nwp-mixture', type=float, default=0.0,
                       help='The percentage of the normal NWP batches in the whole training batches. Ideally at a micro training iter, all devices should have the same strategy.')
    return parser


def extra_args_provider_difflm(parser):
    """difflm arguments."""
    # common
    group = parser.add_argument_group(title='difflm')
    group.add_argument('--debug', action='store_true',
                       help='Whether to run in debug mode.')
    group.add_argument('--turn-on-debugpy', action='store_true',
                       help='Whether to turn on debugpy.')
    group.add_argument('--detect-anomaly', action='store_true',
                       help='Whether to detect anomaly.')
    group.add_argument('--model-running-mode', type=str, default="difflm", choices=["difflm", "difflm-conditional", "vanilla", "test-forward"],
                       help='Running mode, such as difflm.')
    group.add_argument('--base-model', type=str, default="Qwen_2", choices=["vanilla", "Qwen_2"],
                       help='Use Qwen 2 series model by default (including Qwen 2.5, Qwen 2.5 Math, etc.). This only specifies the model series, instead of the specific model name.')
    group.add_argument('--mask-token', type=int, default=151656, # 151656 is the <|video_pad|> token of Qwen 2.5
                       help='Mask token to use for the diffusion lm.')
    group.add_argument('--attention-mask-type', type=str, default='causal_bottom_right', choices=["causal_bottom_right", "no_mask", "block_causal"],
                       help='Attention mask type to use for the diffusion lm.')
    group.add_argument('--core-attn-implementation', type=str, default='TEDotProductAttention', choices=["TEDotProductAttention", "FlexAttention"],
                       help='Core attention implementation to use for the diffusion lm.')
    group.add_argument('--curriculum-seq-lengths', type=str, default="512,1024,2048,4096",
                       help='Sequence lengths to use to train the diffusion lm in curriculum.')
    group.add_argument('--diffusion-span-length', type=int, default=512,
                       help='Length for the diffusion span in for difflm-conditional.')
    group.add_argument('--curriculum-stage-ratios', type=str, default="0.5,0.2,0.2,0.1",
                       help='Ratio for the curriculum stages.')
    group.add_argument('--normal-nwp-mixture', type=float, default=0.1,
                       help='A value between 0 and 1. The percentage of the normal NWP batches in the whole training batches. Ideally at a micro training iter, all devices should have the same strategy.')
    group.add_argument('--use-difflm-validation', action='store_true', 
                       help='Whether to add a diffusion validation in addition to the original NWP validation.')
    return parser