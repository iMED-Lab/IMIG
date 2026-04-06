#!/bin/bash
# Launch multi-modal retinal disease diagnosis training with DDP (2 GPUs)
python -m torch.distributed.launch --nproc_per_node=2 --master_port=21676 --use_env main.py \
    -gpuid='0,1' \
    -m=0.1 \
    -last_layer_fixed=False \
    -subtractive_margin=False \
    -using_deform=False \
    -topk_k=1 \
    -num_prototypes=105 \
    -incorrect_class_connection=-0.5 \
    -deformable_conv_hidden_channels=128 \
    -rand_seed=1
