"""
Training and evaluation routines for the multi-modal retinal disease diagnosis model.

Loss functions (Section "Loss function" in paper):
  - L_BCE (Eq. 5): Binary cross-entropy for multi-label classification
  - L_T (Eq. 2): Typicality loss (feature library constraint)  
  - L_O (Eq. 3): Orthogonality loss (feature diversity constraint)
  - L_P (Eq. 4): Prototype alignment loss (cross-modal shared feature constraint)

Training strategy (Section "Training strategy" in paper):
  Stage 1: Pre-train FFA branch independently
  Stage 2: Pre-train CFP branch with FFA branch frozen, train shared projections
  Stage 3: Joint fine-tuning of all modules
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, accuracy_score, hamming_loss,
                             precision_score, recall_score, f1_score)

from conf import config


def compute_metrics(true_labels, pred_probs):
    """Compute multi-label classification metrics."""
    true_labels = np.array(true_labels)
    pred_probs = np.array(pred_probs)

    val_roc_auc = roc_auc_score(true_labels, pred_probs, average='micro')
    pred = np.where(pred_probs >= 0.5, 1, 0)
    val_hamming_loss = hamming_loss(true_labels, pred)
    val_precision = precision_score(true_labels, pred, average='samples')
    val_recall = recall_score(true_labels, pred, average='samples')
    val_f1 = f1_score(true_labels, pred, average='samples')

    return val_roc_auc, val_precision, val_recall, val_f1, val_hamming_loss


def loss_func_sim(a, b, margin=0.5):
    """Prototype alignment loss L_P (Eq. 4 in the paper).
    
    Contrastive loss that increases similarity between corresponding CFP-FFA
    prototype pairs while decreasing similarity between non-corresponding pairs.
    
    Args:
        a: projected CFP prototypes, shape (num_classes, prototypes_per_class, dim)
        b: projected FFA prototypes, shape (num_classes, prototypes_per_class, dim)
    """
    # Positive pairs: corresponding prototypes (diagonal elements)
    positive_pairs = torch.exp(F.cosine_similarity(a, b, dim=-1))

    # Negative pairs: all non-corresponding combinations within each class
    a_expanded = a.unsqueeze(2).expand(-1, -1, b.size(1), -1)
    b_expanded = b.unsqueeze(1).expand(-1, a.size(1), -1, -1)
    negative_pairs = torch.exp(F.cosine_similarity(a_expanded, b_expanded, dim=-1))

    # Mask out diagonal (positive) entries
    identity_matrix = torch.eye(negative_pairs.shape[-1]).unsqueeze(0).repeat(
        negative_pairs.shape[0], 1, 1).cuda()
    negative_pairs = torch.where(identity_matrix.bool(),
                                 torch.zeros_like(negative_pairs),
                                 negative_pairs)

    # InfoNCE-style contrastive loss
    positive_prob = positive_pairs / (positive_pairs + negative_pairs.mean(dim=-1))
    loss = -torch.log(positive_prob)
    loss = loss.mean(dim=1)

    return loss.mean()


def _train_or_test_multi(model, dataloader, optimizer=None, class_specific=True,
                         use_l1_mask=True, coefs=None, log=print,
                         subtractive_margin=True, use_ortho_loss=False,
                         img_marker='Multi', pretrain=None):
    """Core training/evaluation loop for the multi-modal model.
    
    Args:
        model: DDP-wrapped MultiModel
        dataloader: training or test dataloader
        optimizer: optimizer (None for evaluation)
        pretrain: training stage indicator
            - 'FFA': Stage 1 - pre-train FFA branch only
            - 'CFP': Stage 2 - pre-train CFP branch with cross-modal alignment
            - None:  Stage 3 - joint fine-tuning
    """
    is_train = optimizer is not None
    start = time.time()
    n_batches = 0
    total_cross_entropy = 0
    total_distance_loss = 0
    distance_proto_prototype = 0
    total_ortho_loss_cfp = 0
    total_ortho_loss_ffa = 0

    true_labels_val = []
    pred_probs_val = []
    pred_probs_val_cfp = []
    pred_probs_val_ffa = []
    pred_probs_val_avg = []

    for i, (items, label, _, _, label_sparse) in enumerate(dataloader):
        inputs_FA, inputs_CFP = items
        target = label.cuda()

        input_CFP = inputs_CFP[0].cuda()
        input_FFA = [inputs_FA[i].cuda() for i in range(len(inputs_FA))]

        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # Forward pass through dual-branch model
            if pretrain is None:
                outputs, additional_returns = model(
                    input_CFP, input_FFA, is_train=is_train,
                    prototypes_of_wrong_class=None, incpmplete=True)
            else:
                outputs, additional_returns = model(
                    input_CFP, input_FFA, is_train=is_train,
                    prototypes_of_wrong_class=None)

            output_CFP = outputs[0]
            output_FFA = outputs[1]
            output_fused = outputs[2]
            output_average = 0.7 * output_CFP + 0.3 * output_FFA

            activations_common = additional_returns[2]  # Cross-modal completed activations
            activations_FFA = additional_returns[3]      # Direct FFA activations
            projected_CFP = additional_returns[4]         # Projected CFP prototypes
            projected_FFA = additional_returns[5]         # Projected FFA prototypes

            # L_BCE: Binary cross-entropy losses for each branch and fused output (Eq. 5)
            cross_entropy_cfp = F.binary_cross_entropy_with_logits(output_CFP, target)
            cross_entropy_ffa = F.binary_cross_entropy_with_logits(output_FFA, target)
            cross_entropy_fused = F.binary_cross_entropy_with_logits(output_fused, target)

            # Activation distance: alignment between completed and direct FFA activations
            distance_activation = F.l1_loss(activations_common, activations_FFA)

            # L_P: Cross-modal prototype alignment loss (Eq. 4)
            distance_prototype = loss_func_sim(projected_CFP, projected_FFA)

            # Collect predictions for metric computation
            true_labels_val.extend(label.cpu().numpy())
            pred_probs_val.extend(torch.sigmoid(output_fused).detach().cpu().numpy())
            pred_probs_val_cfp.extend(torch.sigmoid(output_CFP).detach().cpu().numpy())
            pred_probs_val_ffa.extend(torch.sigmoid(output_FFA).detach().cpu().numpy())
            pred_probs_val_avg.extend(torch.sigmoid(output_average).detach().cpu().numpy())

            n_batches += 1
            total_cross_entropy += cross_entropy_fused.item()
            total_distance_loss += distance_activation.item()
            distance_proto_prototype += distance_prototype

        # Compute total loss and backpropagate
        if is_train:
            if pretrain is None:
                # Stage 3: joint training with all losses
                loss = (cross_entropy_cfp + cross_entropy_ffa
                        + distance_activation + 0.1 * distance_prototype
                        + 5 * cross_entropy_fused)
            elif pretrain == 'FFA':
                # Stage 1: FFA branch only
                loss = cross_entropy_ffa
            elif pretrain == 'CFP':
                # Stage 2: CFP branch + cross-modal alignment
                loss = (cross_entropy_cfp + cross_entropy_fused
                        + 0.1 * distance_prototype + distance_activation)
            else:
                print('Loss cannot be calculated: unknown pretrain stage.')
                return

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        del inputs_FA, inputs_CFP, target, outputs, additional_returns
        del cross_entropy_cfp, cross_entropy_ffa, cross_entropy_fused
        del distance_activation, distance_prototype
        if is_train:
            del loss

    # Compute and log evaluation metrics
    auc_fused, pre, recall, f1_fused, hl = compute_metrics(true_labels_val, pred_probs_val)
    auc_cfp, pre_cfp, recall_cfp, f1_cfp, hl_cfp = compute_metrics(true_labels_val, pred_probs_val_cfp)
    auc_ffa, pre_ffa, recall_ffa, f1_ffa, hl_ffa = compute_metrics(true_labels_val, pred_probs_val_ffa)
    auc_avg, _, _, f1_avg, hl_avg = compute_metrics(true_labels_val, pred_probs_val_avg)

    end = time.time()
    log('\ttime: \t{0}'.format(end - start))
    if use_ortho_loss:
        log('\tUsing ortho loss')
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tdistance_activation loss: \t{0}'.format(total_distance_loss / n_batches))
    log('\tdistance_prototype loss: \t{0}'.format(distance_proto_prototype / n_batches))

    log('\tpre: \t\t{0}%'.format(pre))
    log('\tpre cfp: \t\t{0}%'.format(pre_cfp))
    log('\tpre ffa: \t\t{0}%'.format(pre_ffa))

    log('\trecall: \t\t{0}%'.format(recall))
    log('\trecall cfp: \t\t{0}%'.format(recall_cfp))
    log('\trecall ffa: \t\t{0}%'.format(recall_ffa))

    log('\tF1_Score: \t\t{0}%'.format(f1_fused))
    log('\tF1_Score_cfp: \t\t{0}%'.format(f1_cfp))
    log('\tF1_Score_ffa: \t\t{0}%'.format(f1_ffa))
    log('\tF1_Score_avg: \t\t{0}%'.format(f1_avg))

    log('\tAUC: \t\t{0}%'.format(auc_fused))
    log('\tAUC_cfp: \t\t{0}%'.format(auc_cfp))
    log('\tAUC_ffa: \t\t{0}%'.format(auc_ffa))
    log('\tAUC_avg: \t\t{0}%'.format(auc_avg))

    log('\tHamming loss: \t\t{0}%'.format(hl))
    log('\tHamming loss cfp: \t\t{0}%'.format(hl_cfp))
    log('\tHamming loss ffa: \t\t{0}%'.format(hl_ffa))

    # Return metrics and raw predictions based on training stage
    if pretrain == 'FFA':
        return f1_ffa, [true_labels_val, pred_probs_val_cfp, pred_probs_val_ffa, pred_probs_val_avg]
    elif pretrain == 'CFP':
        return f1_cfp, [true_labels_val, pred_probs_val_cfp, pred_probs_val_ffa, pred_probs_val_avg]
    else:
        return f1_fused, [true_labels_val, pred_probs_val_cfp, pred_probs_val_ffa, pred_probs_val]


# ---- Public API ----

def train(model, dataloader, optimizer, class_specific=False, coefs=None,
          log=print, subtractive_margin=True, use_ortho_loss=False,
          img_marker='CFP', pretrianFFA=None):
    """Train for one epoch."""
    assert optimizer is not None
    log('\ttrain')
    model.train()
    return _train_or_test_multi(
        model=model, dataloader=dataloader, optimizer=optimizer,
        class_specific=class_specific, coefs=coefs, log=log,
        subtractive_margin=subtractive_margin, use_ortho_loss=use_ortho_loss,
        img_marker=img_marker, pretrain=pretrianFFA)


def test(model, dataloader, class_specific=False, log=print,
         subtractive_margin=True, img_marker='CFP', pretrianFFA=None):
    """Evaluate on a dataset."""
    log('\ttest')
    model.eval()
    return _train_or_test_multi(
        model=model, dataloader=dataloader, optimizer=None,
        class_specific=class_specific, log=log,
        subtractive_margin=subtractive_margin,
        img_marker=img_marker, pretrain=pretrianFFA)


# ---- Parameter freezing schedules (Section "Training strategy") ----

def last_only(model, log=print, last_layer_fixed=True, img_marker=None, pretrain=None):
    """Freeze everything except the classification layers (last-layer fine-tuning)."""
    if img_marker == 'multi':
        for p in model.module.CFP_branch.features.parameters():
            p.requires_grad = False
        for p in model.module.CFP_branch.add_on_layers.parameters():
            p.requires_grad = False
        model.module.CFP_branch.prototype_vectors.requires_grad = False
        for p in model.module.CFP_branch.conv_offset.parameters():
            p.requires_grad = False
        for p in model.module.CFP_branch.last_layer.parameters():
            p.requires_grad = not last_layer_fixed

        if pretrain == 'CFP':
            # Keep FFA branch completely frozen during CFP pre-training
            for p in model.module.FFA_branch.parameters():
                p.requires_grad = False
        else:
            for p in model.module.FFA_branch.features.parameters():
                p.requires_grad = False
            for p in model.module.FFA_branch.add_on_layers.parameters():
                p.requires_grad = True
            model.module.FFA_branch.prototype_vectors.requires_grad = True
            for p in model.module.FFA_branch.last_layer.parameters():
                p.requires_grad = not last_layer_fixed

        for p in model.module.FFA_branch.conv_offset.parameters():
            p.requires_grad = False
        for p in model.module.last_layer_multi.parameters():
            p.requires_grad = not last_layer_fixed
        for p in model.module.projection_CFP.parameters():
            p.requires_grad = False
        for p in model.module.projection_FFA.parameters():
            p.requires_grad = False
        for p in model.module.gate_CFP.parameters():
            p.requires_grad = False
        for p in model.module.gate_FFA.parameters():
            p.requires_grad = False

    log('\tlast layer')


def warm_only(model, log=print, last_layer_fixed=True, img_marker=None, pretrain=None):
    """Warm-up: train only add-on layers and prototype vectors (backbone frozen)."""
    if img_marker == 'multi':
        # CFP branch: train add-on layers and prototypes
        for p in model.module.CFP_branch.features.parameters():
            p.requires_grad = False
        for p in model.module.CFP_branch.add_on_layers.parameters():
            p.requires_grad = True
        model.module.CFP_branch.prototype_vectors.requires_grad = True
        for p in model.module.CFP_branch.conv_offset.parameters():
            p.requires_grad = False
        for p in model.module.CFP_branch.last_layer.parameters():
            p.requires_grad = not last_layer_fixed

        if pretrain == 'CFP':
            # FFA branch frozen during CFP pre-training stage
            for p in model.module.FFA_branch.parameters():
                p.requires_grad = False
        else:
            for p in model.module.FFA_branch.features.parameters():
                p.requires_grad = False
            for p in model.module.FFA_branch.add_on_layers.parameters():
                p.requires_grad = True
            model.module.FFA_branch.prototype_vectors.requires_grad = True
            for p in model.module.FFA_branch.last_layer.parameters():
                p.requires_grad = not last_layer_fixed

        for p in model.module.FFA_branch.conv_offset.parameters():
            p.requires_grad = False
        for p in model.module.last_layer_multi.parameters():
            p.requires_grad = not last_layer_fixed

        # Projection and gating frozen during warm-up
        for p in model.module.projection_CFP.parameters():
            p.requires_grad = False
        for p in model.module.projection_FFA.parameters():
            p.requires_grad = False
        for p in model.module.gate_CFP.parameters():
            p.requires_grad = False
        for p in model.module.gate_FFA.parameters():
            p.requires_grad = False

    log('\twarm')


def joint(model, log=print, last_layer_fixed=True, img_marker=None, pretrain=None):
    """Joint training: unfreeze add-on layers, prototypes, projections, and gates."""
    if img_marker == 'multi':
        for p in model.module.CFP_branch.features.parameters():
            p.requires_grad = False
        for p in model.module.CFP_branch.add_on_layers.parameters():
            p.requires_grad = True
        model.module.CFP_branch.prototype_vectors.requires_grad = True
        for p in model.module.CFP_branch.conv_offset.parameters():
            p.requires_grad = False
        for p in model.module.CFP_branch.last_layer.parameters():
            p.requires_grad = not last_layer_fixed

        if pretrain == 'CFP':
            for p in model.module.FFA_branch.parameters():
                p.requires_grad = False
        else:
            for p in model.module.FFA_branch.features.parameters():
                p.requires_grad = False
            for p in model.module.FFA_branch.add_on_layers.parameters():
                p.requires_grad = True
            model.module.FFA_branch.prototype_vectors.requires_grad = True
            for p in model.module.FFA_branch.last_layer.parameters():
                p.requires_grad = not last_layer_fixed

        for p in model.module.FFA_branch.conv_offset.parameters():
            p.requires_grad = False
        for p in model.module.last_layer_multi.parameters():
            p.requires_grad = not last_layer_fixed

        # Enable projection and gating training during joint stage
        for p in model.module.projection_CFP.parameters():
            p.requires_grad = True
        for p in model.module.projection_FFA.parameters():
            p.requires_grad = True
        for p in model.module.gate_CFP.parameters():
            p.requires_grad = True
        for p in model.module.gate_FFA.parameters():
            p.requires_grad = True

    log('\tjoint')
