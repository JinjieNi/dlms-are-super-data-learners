cd $PROJECT_DIR
source "envs/.env"

# todo: replace with the stride we want. 
# Now it will convert all checkpoints (skip 0 ckpt every stride).
export SKIP_K_CHECKPOINTS_TO_CONVERT=0

# todo: How many nodes we trained the models with, how many gpus we need to convert the checkpoints. 
# E.g., if we trained the model with 16 nodes, then we need 2 nodes * 8 gpus = 16 gpus in total to convert the ckpt. 
# Make sure the node number and gpus per node specify here is the same as we actually launched.
export GPUS_PER_NODE_CONVERT_CKPT=1
export NUM_NODES_CONVERT_CKPT=1

bash examples/dlm_training/dlm_pretrain_1.7b_test.sh convert_ckpt