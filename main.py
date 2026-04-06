"""
Main training script for the incomplete multi-modal retinal disease diagnosis model.

Implements the three-stage training strategy described in the paper:
  Stage 1: Pre-train FFA branch (feature extractor + prototype library + classifier)
  Stage 2: Pre-train CFP branch with FFA frozen (+ shared projection layers)
  Stage 3: Joint fine-tuning of all modules

Usage:
  python -m torch.distributed.launch --nproc_per_node=2 main.py -gpuid='0,1' ...
  (See run.sh for full example)
"""

import os
import shutil
import argparse
import re

import numpy as np
import torch
import torch.utils.data as data
import torch.distributed as dist
import torchvision.transforms as transforms
from torch import optim
from sklearn.metrics import roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt

from conf import config
from helpers import makedir
import models.model as model
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std
from settings import (img_size, experiment_run, base_architecture, num_classes,
                      img_marker, train_batch_size, test_batch_size,
                      joint_optimizer_lrs, joint_lr_step_size,
                      warm_optimizer_lrs, last_layer_optimizer_lr,
                      coefs, num_warm_epochs, num_train_epochs,
                      push_epochs, num_secondary_warm_epochs, push_start,
                      subtractive_margin)


def plot_roc_curve(test_labels, test_predictions, disease_labels, title='ROC Curve'):
    """Plot per-disease and averaged ROC curves."""
    roc_auc_micro = roc_auc_score(test_labels, test_predictions, average='micro')
    fpr_micro, tpr_micro, _ = roc_curve(test_labels.ravel(), test_predictions.ravel())

    fpr_macro = dict()
    tpr_macro = dict()
    roc_auc_macro = dict()
    for i in range(test_labels.shape[1]):
        fpr_macro[i], tpr_macro[i], _ = roc_curve(test_labels[:, i], test_predictions[:, i])
        roc_auc_macro[i] = auc(fpr_macro[i], tpr_macro[i])

    fpr_macro_grid = np.linspace(0.0, 1.0, test_labels.shape[0])
    mean_tpr_macro = np.zeros_like(fpr_macro_grid)
    for i in range(test_labels.shape[1]):
        mean_tpr_macro += np.interp(fpr_macro_grid, fpr_macro[i], tpr_macro[i])
    mean_tpr_macro /= test_labels.shape[1]
    fpr_macro["avg"] = fpr_macro_grid
    tpr_macro["avg"] = mean_tpr_macro
    roc_auc_macro["avg"] = auc(fpr_macro["avg"], tpr_macro["avg"])

    markers = ['', 's', '^', 'o', 'd', 'x', '*', 'P', 'X', 'h', 'v']

    plt.figure(figsize=(12, 8))
    for i in range(test_labels.shape[1]):
        marker_idx = int(i / 10)
        plt.plot(fpr_macro[i], tpr_macro[i], lw=1.5,
                 marker=markers[marker_idx], markersize=3,
                 label='ROC curve for {} (area = {:.2f})'.format(disease_labels[i], roc_auc_macro[i]))
        log('ROC curve for {} (area = {:.2f})'.format(disease_labels[i], roc_auc_macro[i]))

    plt.plot(fpr_micro, tpr_micro, color='darkorange', lw=1.5, linestyle='--',
             label='Micro-average ROC curve (area = {:.2f})'.format(roc_auc_micro))
    log('Micro-average ROC curve (area = {:.2f})'.format(roc_auc_micro))
    plt.plot(fpr_macro["avg"], tpr_macro["avg"], color='navy', lw=1.5, linestyle='--',
             label='Macro-average ROC curve (area = {:.2f})'.format(roc_auc_macro["avg"]))
    log('Macro-average ROC curve (area = {:.2f})'.format(roc_auc_macro["avg"]))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.subplots_adjust(right=0.75)

    return roc_auc_macro["avg"]


def build_optimizer_multi(PPNet, lr_config, mode='joint'):
    """Build optimizer with per-module learning rates for the multi-modal model.
    
    Args:
        mode: 'warm' (add-on + prototypes only), 'joint' (all trainable modules),
              'last_layer' (classifiers only)
    """
    if mode == 'warm':
        specs = [
            {'params': PPNet.module.CFP_branch.add_on_layers.parameters(),
             'lr': lr_config['add_on_layers'], 'weight_decay': 1e-3},
            {'params': PPNet.module.CFP_branch.prototype_vectors,
             'lr': lr_config['prototype_vectors']},
            {'params': PPNet.module.FFA_branch.add_on_layers.parameters(),
             'lr': lr_config['add_on_layers'], 'weight_decay': 1e-3},
            {'params': PPNet.module.FFA_branch.prototype_vectors,
             'lr': lr_config['prototype_vectors']},
        ]
    elif mode == 'joint':
        specs = [
            {'params': PPNet.module.CFP_branch.features.parameters(),
             'lr': lr_config['features'], 'weight_decay': 1e-3},
            {'params': PPNet.module.CFP_branch.add_on_layers.parameters(),
             'lr': lr_config['add_on_layers'], 'weight_decay': 1e-3},
            {'params': PPNet.module.CFP_branch.prototype_vectors,
             'lr': lr_config['prototype_vectors']},
            {'params': PPNet.module.CFP_branch.conv_offset.parameters(),
             'lr': lr_config['conv_offset']},
            {'params': PPNet.module.CFP_branch.last_layer.parameters(),
             'lr': lr_config['joint_last_layer_lr']},
            {'params': PPNet.module.FFA_branch.features.parameters(),
             'lr': lr_config['features'], 'weight_decay': 1e-3},
            {'params': PPNet.module.FFA_branch.add_on_layers.parameters(),
             'lr': lr_config['add_on_layers'], 'weight_decay': 1e-3},
            {'params': PPNet.module.FFA_branch.prototype_vectors,
             'lr': lr_config['prototype_vectors']},
            {'params': PPNet.module.FFA_branch.conv_offset.parameters(),
             'lr': lr_config['conv_offset']},
            {'params': PPNet.module.FFA_branch.last_layer.parameters(),
             'lr': lr_config['joint_last_layer_lr']},
            {'params': PPNet.module.last_layer_multi.parameters(),
             'lr': lr_config['joint_last_layer_lr']},
            {'params': PPNet.module.projection_CFP.parameters(),
             'lr': lr_config['features']},
            {'params': PPNet.module.projection_FFA.parameters(),
             'lr': lr_config['features']},
            {'params': PPNet.module.gate_CFP.parameters(),
             'lr': lr_config['features']},
            {'params': PPNet.module.gate_FFA.parameters(),
             'lr': lr_config['features']},
        ]
    elif mode == 'last_layer':
        specs = [
            {'params': PPNet.module.CFP_branch.last_layer.parameters(),
             'lr': last_layer_optimizer_lr},
            {'params': PPNet.module.FFA_branch.last_layer.parameters(),
             'lr': last_layer_optimizer_lr},
        ]
    else:
        raise ValueError(f"Unknown optimizer mode: {mode}")

    return torch.optim.Adam(specs)


def save_roc_curves(PPNet, test_loader, disease_labels, model_dir, epoch, log):
    """Evaluate and save ROC curves for CFP, FFA, and averaged predictions."""
    accu, result_pro = tnt.test(
        model=PPNet, dataloader=test_loader,
        class_specific=True, log=log,
        subtractive_margin=subtractive_margin, img_marker=img_marker)

    true_labels = np.array(result_pro[0])
    pred_cfp = np.array(result_pro[1])
    pred_ffa = np.array(result_pro[2])
    pred_avg = np.array(result_pro[3])

    savePath = os.path.join(model_dir, 'figure')
    makedir(savePath)

    for pred, suffix in [(pred_cfp, 'CFP'), (pred_ffa, 'FFA'), (pred_avg, 'AVG')]:
        plot_roc_curve(true_labels, pred, disease_labels, title='ROC')
        plt.savefig(os.path.join(savePath, f'roc_curve_epoch_{epoch+1}_{suffix}.png'))
        plt.clf()

    return accu, result_pro


def run_push_finetune(PPNet, train_loader, test_loader, last_layer_optimizer,
                      class_specific, model_dir, epoch, log, pretrain_stage):
    """Fine-tune last layer after prototype pushing."""
    if not last_layer_fixed_flag:
        tnt.last_only(model=PPNet, log=log, last_layer_fixed=last_layer_fixed_flag,
                      img_marker=img_marker, pretrain=pretrain_stage)
        suffix = 'FFApush' if pretrain_stage == 'FFA' else 'push'
        for i in range(10 if pretrain_stage else 20):
            log('iteration: \t{0}'.format(i))
            _ = tnt.train(model=PPNet, dataloader=train_loader,
                         optimizer=last_layer_optimizer,
                         class_specific=class_specific, coefs=coefs, log=log,
                         subtractive_margin=subtractive_margin,
                         img_marker=img_marker, pretrianFFA=pretrain_stage)
            accu, _ = tnt.test(model=PPNet, dataloader=test_loader,
                              class_specific=class_specific, log=log,
                              img_marker=img_marker, pretrianFFA=pretrain_stage)
            save.save_model_w_condition(
                model=PPNet.module, model_dir=model_dir,
                model_name=f'{epoch}_{i}{suffix}', accu=accu,
                target_accu=0.70, log=log)


# ============================================================================
# Argument parsing
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-m', nargs=1, type=float, default=[1])
parser.add_argument('-last_layer_fixed', nargs=1, type=str, default=[True])
parser.add_argument('-subtractive_margin', nargs=1, type=str, default=[True])
parser.add_argument('-using_deform', nargs=1, type=str, default=[False])
parser.add_argument('-topk_k', nargs=1, type=int, default=[1])
parser.add_argument('-deformable_conv_hidden_channels', nargs=1, type=int, default=[1])
parser.add_argument('-num_prototypes', nargs=1, type=int, default=[2000])
parser.add_argument('-dilation', nargs=1, type=float, default=2)
parser.add_argument('-incorrect_class_connection', nargs=1, type=float, default=[0])
parser.add_argument('-rand_seed', nargs=1, type=int, default=[20])

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
m = args.m[0]
rand_seed = args.rand_seed[0]
last_layer_fixed_flag = args.last_layer_fixed[0] == 'True'
using_deform = args.using_deform[0] == 'True'
topk_k = args.topk_k[0]
deformable_conv_hidden_channels = args.deformable_conv_hidden_channels[0]
num_prototypes = args.num_prototypes[0]
incorrect_class_connection = args.incorrect_class_connection[0]

np.random.seed(rand_seed)
torch.manual_seed(rand_seed)
print(f"Experiment: {experiment_run}, Random seed: {rand_seed}")
print(f"num_prototypes: {num_prototypes}, topk_k: {topk_k}")
print(f"using_deform: {using_deform}, img_marker: {img_marker}")

# ============================================================================
# Model architecture configuration
# ============================================================================

if num_prototypes is None:
    num_prototypes = 1200

# ConvNeXt-Base: 1024-dim features, 1x1 spatial
prototype_shape = (num_prototypes, 1024, 1, 1)
add_on_layers_type = 'identity'
print("Add on layers type:", add_on_layers_type)

# ============================================================================
# Directory setup and logging
# ============================================================================

model_dir = './saved_models/' + experiment_run + '/'
makedir(model_dir)

# Save source files for reproducibility
for src_file in ['main.py', 'settings.py', 'conf.py', 'train_and_test.py']:
    if os.path.exists(src_file):
        shutil.copy(src=os.path.join(os.getcwd(), src_file), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
makedir(os.path.join(model_dir, 'img'))

normalize = transforms.Normalize(mean=mean, std=std)
log("{} classes".format(num_classes))

disease_labels = config.disease_dict

# ============================================================================
# Distributed data parallel setup
# ============================================================================

dist.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device_id = local_rank % torch.cuda.device_count()

# ============================================================================
# Data loaders
# ============================================================================

train_dataset = config.dataset_train
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = data.DataLoader(
    train_dataset, batch_size=config.batchSize,
    sampler=train_sampler, pin_memory=False)
train_push_loader = config.dataloader_train_push
test_loader = config.dataloader_test

log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(config.batchSize))

# ============================================================================
# Model construction
# ============================================================================

PPNet = model.construct_MultiModel(
    base_architecture=base_architecture, pretrained=True, img_size=img_size,
    prototype_shape=prototype_shape, num_classes=num_classes,
    topk_k=topk_k, m=m, add_on_layers_type=add_on_layers_type,
    using_deform=using_deform,
    incorrect_class_connection=incorrect_class_connection,
    deformable_conv_hidden_channels=deformable_conv_hidden_channels,
    prototype_dilation=2, marker=img_marker)

PPNet = PPNet.cuda()
PPNet = torch.nn.parallel.DistributedDataParallel(
    PPNet, find_unused_parameters=True,
    device_ids=[device_id], output_device=device_id)

class_specific = True

# ============================================================================
# Stage 1: Pre-train FFA branch
# ============================================================================

log('=' * 60)
log('Stage 1: Pre-train FFA branch')
log('=' * 60)

joint_optimizer = optim.Adam(PPNet.parameters(), lr=joint_optimizer_lrs['features'])
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    joint_optimizer, step_size=joint_lr_step_size, gamma=0.2)
warm_optimizer = build_optimizer_multi(PPNet, warm_optimizer_lrs, mode='warm')
last_layer_optimizer = build_optimizer_multi(PPNet, {}, mode='last_layer')

for epoch in range(15):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=PPNet, log=log, last_layer_fixed=last_layer_fixed_flag,
                      img_marker=img_marker, pretrain='FFA')
        _ = tnt.train(model=PPNet, dataloader=train_loader, optimizer=warm_optimizer,
                     class_specific=class_specific, coefs=coefs, log=log,
                     subtractive_margin=subtractive_margin,
                     use_ortho_loss=False, img_marker=img_marker, pretrianFFA='FFA')
    else:
        tnt.joint(model=PPNet, log=log, last_layer_fixed=last_layer_fixed_flag,
                  img_marker=img_marker, pretrain='FFA')
        _ = tnt.train(model=PPNet, dataloader=train_loader, optimizer=joint_optimizer,
                     class_specific=class_specific, coefs=coefs, log=log,
                     subtractive_margin=subtractive_margin,
                     use_ortho_loss=True, img_marker=img_marker, pretrianFFA='FFA')
        joint_lr_scheduler.step()

    accu, _ = tnt.test(model=PPNet, dataloader=test_loader,
                       class_specific=class_specific, log=log,
                       subtractive_margin=subtractive_margin,
                       img_marker=img_marker, pretrianFFA='FFA')
    save.save_model_w_condition(
        model=PPNet.module, model_dir=model_dir,
        model_name=str(epoch) + 'pretrain_FFA', accu=accu,
        target_accu=0.76, log=log)

    if (epoch == push_start and push_start < 20) or (epoch >= push_start and epoch in push_epochs):
        run_push_finetune(PPNet, train_loader, test_loader, last_layer_optimizer,
                         class_specific, model_dir, epoch, log, 'FFA')


# ============================================================================
# Stage 2: Pre-train CFP branch (FFA frozen)
# ============================================================================

log('=' * 60)
log('Stage 2: Pre-train CFP branch (FFA frozen)')
log('=' * 60)

joint_optimizer = optim.Adam(PPNet.parameters(), lr=joint_optimizer_lrs['features'])
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    joint_optimizer, step_size=joint_lr_step_size, gamma=0.2)
warm_optimizer = build_optimizer_multi(PPNet, warm_optimizer_lrs, mode='warm')
last_layer_optimizer = build_optimizer_multi(PPNet, {}, mode='last_layer')

log('Pre-testing before CFP training:')
accu, _ = tnt.test(model=PPNet, dataloader=test_loader,
                   class_specific=class_specific, log=log,
                   subtractive_margin=subtractive_margin,
                   img_marker=img_marker, pretrianFFA='CFP')

for epoch in range(15):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=PPNet, log=log, last_layer_fixed=last_layer_fixed_flag,
                      img_marker=img_marker, pretrain='CFP')
        _ = tnt.train(model=PPNet, dataloader=train_loader, optimizer=warm_optimizer,
                     class_specific=class_specific, coefs=coefs, log=log,
                     subtractive_margin=subtractive_margin,
                     use_ortho_loss=False, img_marker=img_marker, pretrianFFA='CFP')
    else:
        tnt.joint(model=PPNet, log=log, last_layer_fixed=last_layer_fixed_flag,
                  img_marker=img_marker, pretrain='CFP')
        _ = tnt.train(model=PPNet, dataloader=train_loader, optimizer=joint_optimizer,
                     class_specific=class_specific, coefs=coefs, log=log,
                     subtractive_margin=subtractive_margin,
                     use_ortho_loss=True, img_marker=img_marker, pretrianFFA='CFP')
        joint_lr_scheduler.step()

    accu, _ = tnt.test(model=PPNet, dataloader=test_loader,
                       class_specific=class_specific, log=log,
                       subtractive_margin=subtractive_margin,
                       img_marker=img_marker, pretrianFFA='CFP')
    save.save_model_w_condition(
        model=PPNet.module, model_dir=model_dir,
        model_name=str(epoch) + 'pretrain_CFP', accu=accu,
        target_accu=0.75, log=log)

    if (epoch == push_start and push_start < 20) or (epoch >= push_start and epoch in push_epochs):
        run_push_finetune(PPNet, train_loader, test_loader, last_layer_optimizer,
                         class_specific, model_dir, epoch, log, 'CFP')


# ============================================================================
# Stage 3: Joint fine-tuning
# ============================================================================

log('=' * 60)
log('Stage 3: Joint fine-tuning')
log('=' * 60)

joint_optimizer = build_optimizer_multi(PPNet, joint_optimizer_lrs, mode='joint')
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    joint_optimizer, step_size=joint_lr_step_size, gamma=0.2)
warm_optimizer = build_optimizer_multi(PPNet, warm_optimizer_lrs, mode='warm')
last_layer_optimizer = build_optimizer_multi(PPNet, {}, mode='last_layer')

log("joint_optimizer_lrs: " + str(joint_optimizer_lrs))
log("warm_optimizer_lrs: " + str(warm_optimizer_lrs))

for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=PPNet, log=log, last_layer_fixed=last_layer_fixed_flag,
                      img_marker=img_marker)
        _, _ = tnt.train(model=PPNet, dataloader=train_loader, optimizer=warm_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log,
                        subtractive_margin=subtractive_margin,
                        use_ortho_loss=False, img_marker=img_marker)
    else:
        tnt.joint(model=PPNet, log=log, last_layer_fixed=last_layer_fixed_flag,
                  img_marker=img_marker)
        _ = tnt.train(model=PPNet, dataloader=train_loader, optimizer=joint_optimizer,
                     class_specific=class_specific, coefs=coefs, log=log,
                     subtractive_margin=subtractive_margin,
                     use_ortho_loss=True, img_marker=img_marker)
        joint_lr_scheduler.step()

    accu, result_pro = save_roc_curves(
        PPNet, test_loader, disease_labels, model_dir, epoch, log)
    save.save_model_w_condition(
        model=PPNet.module, model_dir=model_dir,
        model_name=str(epoch) + 'nopush', accu=accu,
        target_accu=0.76, log=log)

    if (epoch == push_start and push_start < 20) or (epoch >= push_start and epoch in push_epochs):
        run_push_finetune(PPNet, train_loader, test_loader, last_layer_optimizer,
                         class_specific, model_dir, epoch, log, None)

logclose()
