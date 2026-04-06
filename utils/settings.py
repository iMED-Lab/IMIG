"""
Hyperparameters and training schedule for the incomplete multi-modal learning framework.
"""

base_architecture = 'convnext_base'
img_size = 224
img_marker = 'multi'  # 'multi' for dual-branch CFP+FFA model

experiment_run = '51_testNumber'

data_path = '.'
num_classes = 7  # AMD, ME, VH, HighMyopia, CSC, DR, RVO

train_dir = data_path + '.'
test_dir = data_path + '.'
train_push_dir = data_path + '.'

train_batch_size = 20
test_batch_size = 100
train_push_batch_size = 40

# ---- Optimizer learning rates ----

# Joint training stage: fine-tune all modules together
joint_optimizer_lrs = {
    'features': 1e-4,
    'add_on_layers': 3e-3,
    'prototype_vectors': 3e-3,
    'conv_offset': 1e-4,
    'joint_last_layer_lr': 1e-5
}
joint_lr_step_size = 5

# Warm-up stage: only train add-on layers and prototype vectors
warm_optimizer_lrs = {
    'add_on_layers': 3e-3,
    'prototype_vectors': 3e-3
}

# Secondary warm-up: also unfreeze backbone features
warm_pre_offset_optimizer_lrs = {
    'add_on_layers': 3e-3,
    'prototype_vectors': 3e-3,
    'features': 1e-4
}

last_layer_optimizer_lr = 1e-4
last_layer_fixed = True

# ---- Loss coefficients ----
coefs = {
    'crs_ent': 1,
    'clst': -0.8,
    'sep': 0.08,
    'l1': 1e-2,
    'offset_bias_l2': 8e-1,
    'offset_weight_l2': 8e-1,
    'orthogonality_loss': 0.1
}

subtractive_margin = True

# ---- Training schedule ----
num_train_epochs = 55
num_warm_epochs = 3
num_secondary_warm_epochs = 5
push_start = 11
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
