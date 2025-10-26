def extra_args_provider(parser):
    """difflm arguments."""
    # common
    group = parser.add_argument_group(title='difflm')
    group.add_argument('--debug', action='store_true',
                       help='Whether to run in debug mode.')
    group.add_argument('--turn-on-debugpy', action='store_true',
                       help='Whether to turn on debugpy.')
    group.add_argument('--detect-anomaly', action='store_true',
                       help='Whether to detect anomaly.')
    group.add_argument('--model-running-mode', type=str, default="difflm-noshift", choices=["difflm-noshift", "vanilla", "test-forward"], 
                       help='Running mode, difflm-noshift means vanilla dlm, vanilla means AR.')
    group.add_argument('--base-model', type=str, default="Qwen_2", choices=["vanilla"],
                       help='Use Qwen 2 series model by default (including Qwen 2.5, Qwen 2.5 Math, etc.). This only specifies the model series, instead of the specific model name.')
    group.add_argument('--mask-token', type=int, default=151656, # 151656 is the <|video_pad|> token of Qwen 2.5
                       help='Mask token to use for the diffusion lm.')
    group.add_argument('--attention-mask-type', type=str, default='causal_bottom_right', choices=["causal_bottom_right", "no_mask", "block_causal", "causal"],
                       help='Attention mask type to use for the diffusion lm.')
    group.add_argument('--core-attn-implementation', type=str, default='TEDotProductAttention', choices=["TEDotProductAttention", "FlexAttention"],
                       help='Core attention implementation to use for the diffusion lm.')
    group.add_argument('--non-special-vocab-size', type=int, default=100255,
                       help='The size of the non-special vocab. This is used to replace the model input for uniform tokens.')
    group.add_argument('--cut-off-varlen-to-seqlen', action='store_true',
                       help='Whether to cut off the variable length sequence to the max sequence length. This is only used when we use document packing and intra-doc attention mask.')
    group.add_argument('--checkpoint-steps', type=str, default=None,
                       help='The steps to save the checkpoint, separated by commas.')
    group.add_argument('--difflm-varilen-prob', type=float, default=0.01,
                       help='The probability of using variable length data for the diffusion lm.')
    
    # eval arguments
    group.add_argument('--use-on-the-fly-eval', action='store_true',
                       help='Whether to use on-the-fly eval.')
    return parser
