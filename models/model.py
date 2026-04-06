"""
Multi-modal prototype-based network for incomplete multi-modal retinal disease diagnosis.

Architecture overview (corresponds to paper Section "Framework"):
  - PPNet: Single-branch prototype network (used as CFP branch or FFA branch).
           Extracts features via a ConvNeXt backbone, computes cosine similarity
           between spatial features and a learnable typical feature library (prototypes),
           and classifies based on feature-library activation patterns.
  - MultiModel: Dual-branch model combining CFP_branch and FFA_branch (both PPNet).
           Adds shared-feature projection heads (P_c, P_f) for cross-modal alignment,
           gating modules for feature fusion, and a joint classifier.
           During inference, FFA features are completed via CFP->FFA library indexing.

           The code is partially adapted from the original implementation of https://github.com/Henrymachiyu/This-looks-like-those_ProtoConcepts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convnext_features import convnext_base_featuresCFP, convnext_base_featuresFFA
from utils.receptive_field import compute_proto_layer_rf_info_v2


class PPNet(nn.Module):
    """Single-branch Prototypical Part Network.
    
    Each branch maintains:
      - features: ConvNeXt backbone for spatial feature extraction
      - add_on_layers: optional adaptation layers after backbone
      - prototype_vectors: learnable typical feature library T (M prototypes x D dims)
      - conv_offset: offset prediction network (for optional deformable convolution)
      - last_layer: linear classifier from prototype activations to disease logits
    """

    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, topk_k=1,
                 m=None, init_weights=True, add_on_layers_type='bottleneck',
                 using_deform=True, incorrect_class_connection=-1,
                 deformable_conv_hidden_channels=0, prototype_dilation=2,
                 img_marker='CFP'):

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.m = m
        self.using_deform = using_deform
        self.relu_on_cos = True
        self.incorrect_class_connection = incorrect_class_connection
        self.input_vector_length = 64
        self.n_eps_channels = 2
        self.epsilon_val = 1e-4
        self.prototype_dilation = (prototype_dilation, prototype_dilation)
        self.prototype_padding = (1, 1)
        self.topk_k = topk_k
        self.img_marker = img_marker

        # Allocate prototypes evenly across disease classes
        assert self.num_prototypes % self.num_classes == 0
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)
        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info
        self.features = features

        # Determine the number of output channels from the backbone
        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES') or features_name.startswith('CONV'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('Unsupported backbone architecture')

        # Build add-on layers between backbone and prototype matching
        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(current_in_channels, current_out_channels, kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(current_out_channels, current_out_channels, kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)

        elif add_on_layers_type == 'identity':
            if self.img_marker == 'CFP':
                self.add_on_layers = nn.Sequential(nn.Identity())
            else:
                self.add_on_layers = nn.Sequential(
                    nn.Conv2d(first_add_on_layer_in_channels, first_add_on_layer_in_channels, kernel_size=1),
                    nn.Identity()
                )

        elif add_on_layers_type == 'upsample':
            if self.img_marker == 'CFP':
                self.add_on_layers = nn.Upsample(scale_factor=2, mode='bilinear')
            else:
                self.add_on_layers = nn.Sequential(
                    nn.Conv2d(first_add_on_layer_in_channels, first_add_on_layer_in_channels, kernel_size=1),
                    nn.Upsample(scale_factor=2, mode='bilinear')
                )

        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(first_add_on_layer_in_channels, self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(self.prototype_shape[1], self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
            )

        # Learnable typical feature library (prototypes)
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

        # Offset prediction for deformable convolution
        self.deformable_conv_out_channels = 2 * self.prototype_shape[-1] * self.prototype_shape[-2]
        self.deformable_conv_hidden_channels = deformable_conv_hidden_channels

        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        if not self.deformable_conv_hidden_channels:
            conv_offset_1 = nn.Conv2d(
                self.prototype_shape[-3] + self.n_eps_channels,
                self.deformable_conv_out_channels,
                kernel_size=(self.prototype_shape[-2] + 1, self.prototype_shape[-1] + 1),
                stride=(1, 1), padding=(1, 1), dilation=(1, 1), bias=False)
            self.conv_offset = nn.Sequential(conv_offset_1)
        else:
            conv_offset_1 = nn.Conv2d(
                self.prototype_shape[-3] + self.n_eps_channels,
                self.deformable_conv_hidden_channels,
                kernel_size=(self.prototype_shape[-2] + 2, self.prototype_shape[-1] + 2),
                stride=(1, 1), padding=(1, 1), dilation=(1, 1), bias=True)
            conv_offset_2 = nn.Conv2d(
                self.deformable_conv_hidden_channels,
                self.deformable_conv_out_channels,
                kernel_size=(self.prototype_shape[-2], self.prototype_shape[-1]),
                stride=(1, 1), padding=(1, 1), dilation=(1, 1), bias=True)
            self.conv_offset = nn.Sequential(conv_offset_1, nn.ReLU(), conv_offset_2)

        for p in self.conv_offset.modules():
            if isinstance(p, nn.Conv2d):
                torch.nn.init.zeros_(p.weight)
                if p.bias is not None:
                    torch.nn.init.zeros_(p.bias)

        # Classifier: maps prototype activation vector to disease logits
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        """Extract spatial feature maps from the backbone + add-on layers."""
        if self.img_marker == 'CFP':
            x_feat = self.features(x)
        else:
            # For FFA: stack multi-frame grayscale inputs as pseudo-RGB, average across frames
            x = torch.cat(x, 1).unsqueeze(2)
            x = x.repeat(1, 1, 3, 1, 1)
            b, n, c, h, w = x.shape
            x = x.reshape(b * n, c, h, w)
            x_feat = self.features(x)
            feat_h, feat_w = x_feat.shape[-2], x_feat.shape[-1]
            x_feat = x_feat.reshape(b, n, -1, feat_h, feat_w)
            x_feat = torch.mean(x_feat, 1)
        x = self.add_on_layers(x_feat)
        return x

    def cos_activation(self, x, is_train=True, prototypes_of_wrong_class=None):
        """Compute cosine similarity between spatial features and prototype vectors.
        
        Implements the hyperspherical cosine similarity metric described in the paper:
        features and prototypes are projected onto a unit hypersphere, and similarity
        is computed via normalized dot product (Eq. 1 in the paper).
        """
        input_vector_length = self.input_vector_length
        normalizing_factor = (self.prototype_shape[-2] * self.prototype_shape[-1]) ** 0.5

        # Append epsilon channels to prevent zero vectors
        epsilon_channel_x = torch.ones(x.shape[0], self.n_eps_channels, x.shape[2], x.shape[3]) * self.epsilon_val
        epsilon_channel_x = epsilon_channel_x.cuda()
        epsilon_channel_x.requires_grad = False
        x = torch.cat((x, epsilon_channel_x), -3)

        # Normalize feature patches to fixed length
        x_length = torch.sqrt(torch.sum(torch.square(x), dim=-3) + self.epsilon_val)
        x_length = x_length.view(x_length.size()[0], 1, x_length.size()[1], x_length.size()[2])
        x_normalized = input_vector_length * x / x_length
        x_normalized = x_normalized / normalizing_factor

        # Normalize prototype vectors to unit length
        epsilon_channel_p = torch.ones(
            self.prototype_shape[0], self.n_eps_channels,
            self.prototype_shape[2], self.prototype_shape[3]) * self.epsilon_val
        epsilon_channel_p = epsilon_channel_p.cuda()
        epsilon_channel_p.requires_grad = False
        appended_protos = torch.cat((self.prototype_vectors, epsilon_channel_p), -3)

        prototype_vector_length = torch.sqrt(torch.sum(torch.square(appended_protos), dim=-3) + self.epsilon_val)
        prototype_vector_length = prototype_vector_length.view(
            prototype_vector_length.size()[0], 1,
            prototype_vector_length.size()[1], prototype_vector_length.size()[2])
        normalized_prototypes = appended_protos / (prototype_vector_length + self.epsilon_val)
        normalized_prototypes = normalized_prototypes / normalizing_factor

        # Compute offset (for optional deformable convolution)
        offset = self.conv_offset(x_normalized)

        # Standard convolution-based cosine similarity (deformable conv disabled by default)
        activations_dot = F.conv2d(x_normalized, normalized_prototypes)

        marginless_activations = activations_dot / (input_vector_length * 1.01)

        if self.m is None or not is_train or prototypes_of_wrong_class is None:
            activations = marginless_activations
        else:
            # Subtractive margin for wrong-class prototypes
            wrong_class_margin = prototypes_of_wrong_class * self.m
            wrong_class_margin = wrong_class_margin.view(x.size()[0], self.prototype_vectors.size()[0], 1, 1)
            wrong_class_margin = torch.repeat_interleave(wrong_class_margin, activations_dot.size()[-2], dim=-2)
            wrong_class_margin = torch.repeat_interleave(wrong_class_margin, activations_dot.size()[-1], dim=-1)
            penalized_angles = torch.arccos(activations_dot / (input_vector_length * 1.01)) - wrong_class_margin
            activations = torch.cos(torch.relu(penalized_angles))

        if self.relu_on_cos:
            activations = torch.relu(activations)
            marginless_activations = torch.relu(marginless_activations)

        return activations, marginless_activations

    def prototype_activations(self, x, is_train=True, prototypes_of_wrong_class=None):
        """Compute prototype activations for raw input images."""
        conv_features = self.conv_features(x)
        activations, marginless_activations = self.cos_activation(
            conv_features, is_train=is_train,
            prototypes_of_wrong_class=prototypes_of_wrong_class)
        return activations, [marginless_activations, conv_features]

    def forward(self, x, is_train=True, prototypes_of_wrong_class=None):
        activations, additional_returns = self.prototype_activations(
            x, is_train=is_train, prototypes_of_wrong_class=prototypes_of_wrong_class)
        marginless_activations = additional_returns[0]
        conv_features = additional_returns[1]

        topk_k = self.topk_k if is_train else 1

        # Global max-pooled activation per prototype (Eq. 1: max over spatial locations)
        activations = activations.view(activations.shape[0], activations.shape[1], -1)
        topk_activations, _ = torch.topk(activations, topk_k, dim=-1)
        mean_activations = torch.mean(topk_activations, dim=-1)

        marginless_max_activations = F.max_pool2d(
            marginless_activations,
            kernel_size=(marginless_activations.size()[2], marginless_activations.size()[3]))
        marginless_max_activations = marginless_max_activations.view(-1, self.num_prototypes)

        logits = self.last_layer(mean_activations)
        marginless_logits = self.last_layer(marginless_max_activations)

        return logits, [mean_activations, marginless_logits, conv_features, activations]

    def push_forward(self, x):
        """Forward pass for prototype pushing (visualization)."""
        conv_output = self.conv_features(x)
        _, marginless_activations = self.cos_activation(conv_output)
        return conv_output, marginless_activations

    def get_prototype_orthogonalities(self):
        """Compute orthogonality matrix for the prototype vectors (L_O in the paper, Eq. 3).
        
        Encourages diversity among prototypes within each disease category by
        penalizing non-orthogonal prototype pairs.
        """
        prototype_vector_length = torch.sqrt(
            torch.sum(torch.square(self.prototype_vectors), dim=-3) + self.epsilon_val)
        prototype_vector_length = prototype_vector_length.view(
            prototype_vector_length.size()[0], 1,
            prototype_vector_length.size()[1], prototype_vector_length.size()[2])
        normalized_prototypes = self.prototype_vectors / (prototype_vector_length + self.epsilon_val)

        # Reshape: (num_prototypes_per_class, num_classes * spatial, channels)
        prototype_piece_matrices = normalized_prototypes.view(
            self.num_prototypes_per_class,
            self.num_prototypes // self.num_prototypes_per_class,
            self.prototype_shape[-3],
            self.prototype_shape[-2] * self.prototype_shape[-1])
        prototype_piece_matrices = prototype_piece_matrices.transpose(2, 3).reshape(
            self.num_prototypes_per_class, -1, self.prototype_shape[-3])
        prototype_piece_matrices = prototype_piece_matrices.transpose(1, 2)

        orthogonalities = torch.matmul(prototype_piece_matrices.transpose(-2, -1), prototype_piece_matrices)
        orthogonalities -= torch.eye(
            (self.num_prototypes // self.num_prototypes_per_class) *
            self.prototype_shape[-2] * self.prototype_shape[-1]).cuda()
        return orthogonalities

    def prune_prototypes(self, prototypes_to_prune):
        """Remove specified prototypes from the library."""
        prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))

        self.prototype_vectors = nn.Parameter(
            self.prototype_vectors.data[prototypes_to_keep, ...], requires_grad=True)
        self.prototype_shape = list(self.prototype_vectors.size())
        self.num_prototypes = self.prototype_shape[0]

        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...], requires_grad=False)
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        """Initialize classifier weights: +1 for correct class, negative for incorrect."""
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations
        self.last_layer.weight.data.copy_(
            1.0 * positive_one_weights_locations
            + incorrect_strength * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=self.incorrect_class_connection)

    def __repr__(self):
        return (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        ).format(self.features, self.img_size, self.prototype_shape,
                 self.proto_layer_rf_info, self.num_classes, self.epsilon_val)


class MultiModel(nn.Module):
    """Dual-branch multi-modal model for incomplete multi-modal learning.
    
    Architecture (paper Figure 1):
      - CFP_branch (PPNet): processes color fundus photographs
      - FFA_branch (PPNet): processes fluorescein fundus angiography images
      - projection_CFP / projection_FFA: shared feature projection heads (P_c, P_f)
        that map prototype vectors into a common 128-dim latent space for cross-modal
        alignment (trained with L_P, Eq. 4)
      - gate_CFP / gate_FFA: channel-wise gating modules for adaptive feature fusion
      - last_layer_multi: joint classifier on fused activations
    
    During inference (FFA unavailable):
      CFP features are projected into the shared space and matched against the
      FFA prototype library to reconstruct the missing FFA activation pattern.
    """

    def __init__(self, feature1, feature2, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, topk_k=1,
                 m=None, init_weights=True, add_on_layers_type='bottleneck',
                 using_deform=True, incorrect_class_connection=-1,
                 deformable_conv_hidden_channels=0, prototype_dilation=2,
                 img_marker='CFP'):

        super(MultiModel, self).__init__()

        # Dual branches with independent backbones and prototype libraries
        self.CFP_branch = PPNet(
            features=feature1, img_size=img_size,
            prototype_shape=prototype_shape, proto_layer_rf_info=proto_layer_rf_info,
            num_classes=num_classes, topk_k=topk_k, m=m, init_weights=True,
            add_on_layers_type=add_on_layers_type, using_deform=using_deform,
            incorrect_class_connection=incorrect_class_connection,
            deformable_conv_hidden_channels=deformable_conv_hidden_channels,
            prototype_dilation=prototype_dilation, img_marker='CFP')

        self.FFA_branch = PPNet(
            features=feature2, img_size=img_size,
            prototype_shape=prototype_shape, proto_layer_rf_info=proto_layer_rf_info,
            num_classes=num_classes, topk_k=topk_k, m=m, init_weights=True,
            add_on_layers_type=add_on_layers_type, using_deform=using_deform,
            incorrect_class_connection=incorrect_class_connection,
            deformable_conv_hidden_channels=deformable_conv_hidden_channels,
            prototype_dilation=prototype_dilation, img_marker='FFA')

        # Joint classifier on fused prototype activations
        self.last_layer_multi = nn.Linear(
            self.CFP_branch.num_prototypes, self.CFP_branch.num_classes, bias=False)

        # Shared feature projection heads for cross-modal alignment (128-dim latent space)
        self.projection_CFP = nn.Sequential(
            nn.Linear(self.CFP_branch.prototype_shape[-3], 128),
        )
        self.projection_FFA = nn.Sequential(
            nn.Linear(self.CFP_branch.prototype_shape[-3], 128),
        )

        # Channel-wise gating for adaptive fusion of CFP and completed-FFA activations
        self.gate_CFP = nn.Sequential(
            nn.Conv2d(self.CFP_branch.prototype_shape[0], self.CFP_branch.prototype_shape[0], kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.CFP_branch.prototype_shape[0], self.CFP_branch.prototype_shape[0], kernel_size=1),
            nn.Sigmoid()
        )
        self.gate_FFA = nn.Sequential(
            nn.Conv2d(self.CFP_branch.prototype_shape[0], self.CFP_branch.prototype_shape[0], kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.CFP_branch.prototype_shape[0], self.CFP_branch.prototype_shape[0], kernel_size=1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def forward(self, x1, x2, is_train=True, prototypes_of_wrong_class=None, incpmplete=False):
        """
        Args:
            x1: CFP input image tensor
            x2: FFA input (list of frame tensors)
            is_train: training mode flag
            incpmplete: if True, use cross-modal feature completion (inference mode)
        
        Returns:
            logits: [logits_CFP, logits_FFA, logits_fused]
            features: [max_act_CFP, max_act_FFA, act_common, act_FFA, proj_CFP, proj_FFA]
        """
        # Forward through individual branches
        logits_CFP, [max_activations_CFP, _, conv_features_CFP, activation_CFP] = self.CFP_branch(x1, is_train)
        logits_FFA, [max_activations_FFA, _, conv_features_FFA, activation_FFA] = self.FFA_branch(x2, is_train)

        # Project prototype vectors into shared latent space for cross-modal alignment
        b, dim, h, w = self.CFP_branch.prototype_vectors.shape
        proto_cfp = self.CFP_branch.prototype_vectors.transpose(1, 3).contiguous().reshape(b * w * h, dim)
        proto_ffa = self.FFA_branch.prototype_vectors.transpose(1, 3).contiguous().reshape(b * w * h, dim)
        projected_CFP = self.projection_CFP(proto_cfp)
        projected_FFA = self.projection_FFA(proto_ffa)

        _, dim_new = projected_CFP.shape
        num_class = self.CFP_branch.num_classes
        per_num = int(self.CFP_branch.num_prototypes / self.CFP_branch.num_classes)

        projected_CFP = projected_CFP.reshape(num_class, per_num, dim_new)
        projected_FFA = projected_FFA.reshape(num_class, per_num, dim_new)

        # Cross-modal feature completion: use CFP features to index FFA prototype library
        n = int(activation_CFP.shape[-1] ** 0.5)
        activation_CFP_reshape = activation_CFP.view(activation_CFP.shape[0], activation_CFP.shape[1], n, n)

        gate_cfp = self.gate_CFP(activation_CFP_reshape)

        # Project FFA prototypes and CFP dense features into shared space
        proto_ffa_projected = projected_FFA.contiguous().view(b, h, w, 128).transpose(1, 3)
        f_b, f_d, f_h, f_w = conv_features_CFP.shape
        featureCFP = conv_features_CFP.transpose(1, 3).contiguous().reshape(f_b * f_w * f_h, f_d)
        featureCFP_projected = self.projection_CFP(featureCFP)
        featureCFP_projected_reshaped = featureCFP_projected.contiguous().view(f_b, f_h, f_w, 128).transpose(1, 3)

        # Compute cross-modal activation: CFP features matched against FFA prototypes
        activation_common_before, _ = self.cos_activation_projection(
            featureCFP_projected_reshaped, proto_ffa_projected)
        activation_common = activation_common_before.view(
            activation_common_before.shape[0], activation_common_before.shape[1], -1)
        topk_activations, _ = torch.topk(activation_common, 1, dim=-1)
        mean_activations_common = torch.mean(topk_activations, dim=-1)

        gate_ffa = self.gate_FFA(activation_common_before).squeeze()

        # Fuse CFP activations and completed FFA activations via gating
        fusion_activation = max_activations_CFP * gate_cfp.squeeze() + mean_activations_common * gate_ffa.squeeze()
        logits_mean = self.last_layer_multi(fusion_activation)

        return (
            [logits_CFP, logits_FFA, logits_mean],
            [max_activations_CFP, max_activations_FFA,
             activation_common, activation_FFA,
             projected_CFP, projected_FFA]
        )

    def cos_activation_projection(self, x, proto_project, is_train=True,
                                  prototypes_of_wrong_class=None):
        """Compute cosine similarity in the shared projection space.
        
        Used for cross-modal feature completion: CFP dense features are compared
        against FFA prototype vectors in the shared 128-dim latent space.
        """
        input_vector_length = self.FFA_branch.input_vector_length
        normalizing_factor = (self.FFA_branch.prototype_shape[-2] * self.FFA_branch.prototype_shape[-1]) ** 0.5

        # Append epsilon channels
        epsilon_channel_x = torch.ones(x.shape[0], self.FFA_branch.n_eps_channels, x.shape[2], x.shape[3]) * self.FFA_branch.epsilon_val
        epsilon_channel_x = epsilon_channel_x.cuda()
        epsilon_channel_x.requires_grad = False
        x = torch.cat((x, epsilon_channel_x), -3)

        # Normalize features
        x_length = torch.sqrt(torch.sum(torch.square(x), dim=-3) + self.FFA_branch.epsilon_val)
        x_length = x_length.view(x_length.size()[0], 1, x_length.size()[1], x_length.size()[2])
        x_normalized = input_vector_length * x / x_length
        x_normalized = x_normalized / normalizing_factor

        # Normalize projected prototypes
        epsilon_channel_p = torch.ones(
            self.FFA_branch.prototype_shape[0], self.FFA_branch.n_eps_channels,
            self.FFA_branch.prototype_shape[2], self.FFA_branch.prototype_shape[3]) * self.FFA_branch.epsilon_val
        epsilon_channel_p = epsilon_channel_p.cuda()
        epsilon_channel_p.requires_grad = False
        appended_protos = torch.cat((proto_project, epsilon_channel_p), -3)

        prototype_vector_length = torch.sqrt(torch.sum(torch.square(appended_protos), dim=-3) + self.FFA_branch.epsilon_val)
        prototype_vector_length = prototype_vector_length.view(
            prototype_vector_length.size()[0], 1,
            prototype_vector_length.size()[1], prototype_vector_length.size()[2])
        normalized_prototypes = appended_protos / (prototype_vector_length + self.FFA_branch.epsilon_val)
        normalized_prototypes = normalized_prototypes / normalizing_factor

        activations_dot = F.conv2d(x_normalized, normalized_prototypes)
        marginless_activations = activations_dot / (input_vector_length * 1.01)

        activations = marginless_activations
        if self.FFA_branch.relu_on_cos:
            activations = torch.relu(activations)
            marginless_activations = torch.relu(marginless_activations)

        return activations, marginless_activations

    def set_last_layer_incorrect_connection_multi(self, incorrect_strength):
        """Initialize the joint classifier weights."""
        positive_one_weights_locations = torch.t(self.CFP_branch.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations
        self.last_layer_multi.weight.data.copy_(
            1.0 * positive_one_weights_locations
            + incorrect_strength * negative_one_weights_locations)

    def _initialize_weights(self):
        self.set_last_layer_incorrect_connection_multi(
            incorrect_strength=self.CFP_branch.incorrect_class_connection)


def construct_MultiModel(base_architecture, pretrained=True, img_size=224,
                         prototype_shape=(2000, 512, 1, 1),
                         num_classes=200, topk_k=1, m=None,
                         add_on_layers_type='bottleneck', using_deform=True,
                         incorrect_class_connection=-1,
                         deformable_conv_hidden_channels=128,
                         prototype_dilation=2, marker='CFP'):
    """Factory function to construct the dual-branch MultiModel."""
    features_CFP = convnext_base_featuresCFP(pretrained=pretrained)
    features_FFA = convnext_base_featuresFFA(pretrained=pretrained)

    layer_filter_sizes, layer_strides, layer_paddings = features_CFP.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(
        img_size=img_size,
        layer_filter_sizes=layer_filter_sizes,
        layer_strides=layer_strides,
        layer_paddings=layer_paddings,
        prototype_kernel_size=prototype_shape[2])

    print("prototype_shape:", prototype_shape)

    return MultiModel(
        feature1=features_CFP, feature2=features_FFA,
        img_size=img_size, prototype_shape=prototype_shape,
        proto_layer_rf_info=proto_layer_rf_info,
        num_classes=num_classes, topk_k=topk_k, m=m,
        init_weights=True, add_on_layers_type=add_on_layers_type,
        using_deform=using_deform,
        incorrect_class_connection=incorrect_class_connection,
        deformable_conv_hidden_channels=deformable_conv_hidden_channels,
        prototype_dilation=prototype_dilation, img_marker=marker)
