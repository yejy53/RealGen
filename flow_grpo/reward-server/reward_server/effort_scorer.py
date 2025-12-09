import os
import re
import math
import json
import datetime
import logging
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from sklearn import metrics
from typing import Union
from collections import defaultdict

from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig

from typing import Any, Optional, Tuple, Union, Dict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint
from pytorch_metric_learning import losses
from transformers import CLIPModel, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings, CLIPMLP
logger = logging.getLogger(__name__)



class EffortMoeDetector(nn.Module):
    def __init__(self, config=None, device="cuda"):
        super(EffortMoeDetector, self).__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.rank_per_expert = config.rank_per_expert

        self.moe_lambda_orth = config.moe_lambda_orth
        self.moe_lambda_balance = config.moe_lambda_balance
        self.moe_lambda_sv = config.moe_lambda_sv

        self.moe_lambda_contrastive = 0.0
        self.contrastive_pos_margin = 0.0
        self.contrastive_neg_margin = 1.0

        self.moe_router_hidden_dim = config.moe_router_hidden_dim
        self.top_k = config.moe_top_k

        self.training_mode = "standard" # Default runtime mode, can be changed by set_training_mode
        self.active_expert_idx = None

        # This property is fixed after initialization.
        self.is_hybrid = config.training_mode != 'standard'

        self.device = device

        if self.is_hybrid:
            print("Initializing in HYBRID MoE architectural mode.")
            # Explicitly define which expert is the fixed artifact expert. 
            # Assuming the last expert is the artifact expert.
            self.artifact_expert_idx = config.num_experts - 1
            # The number of experts the router must choose from is the total number minus the fixed one.
            self.num_semantic_experts = config.num_experts - 1
            if self.num_semantic_experts <= 0:
                raise ValueError("num_experts must be at least 2 for the Hybrid MoE model (1 fixed, >=1 semantic).")
            gating_num_experts = self.num_semantic_experts
        else:
            print("Initializing in STANDARD MoE architectural mode.")
            self.artifact_expert_idx = -1  # A value that will never match an expert index
            gating_num_experts = self.num_experts


        pretrained_path = "../RealGen/models/openaiclip-vit-large-patch14-336"

        clip_model = CLIPModel.from_pretrained(pretrained_path)
        vision_config = clip_model.vision_model.config
        self.hidden_size = vision_config.hidden_size
        
        # Calculate rank allocation
        total_rank = vision_config.hidden_size
        residual_rank = self.rank_per_expert

        if residual_rank >= total_rank:
            raise ValueError(
                f"The total rank for experts ({residual_rank}) must be less than the total rank ({total_rank}). "
                f"Please reduce num_experts ({self.num_experts}) or rank_per_expert ({self.rank_per_expert})."
            )
        
        r_main = total_rank - residual_rank
        print(f"Rank allocation: total_rank={total_rank}, r_main={r_main}, num_experts={self.num_experts}, rank_per_expert={self.rank_per_expert}")
        
        # Build the MoE backbone network
        self.embeddings = CLIPVisionEmbeddings(vision_config)
        self.ln_pre = nn.LayerNorm(vision_config.hidden_size)

        self.encoder_layers = nn.ModuleList([
            ViTMoELayer(vision_config, self.num_experts, r_main, self.rank_per_expert, self.artifact_expert_idx) for _ in range(vision_config.num_hidden_layers)
        ])

        self.ln_post = nn.LayerNorm(self.hidden_size, eps=vision_config.layer_norm_eps)
        # self.visual_projection = nn.Linear(vision_config.hidden_size, clip_model.projection_dim, bias=False)

        # self.gating_network = GatingNetwork(
        #     input_dim=self.hidden_size, 
        #     num_experts=self.num_experts, 
        #     hidden_dim=self.moe_router_hidden_dim, 
        #     top_k=self.top_k
        # )

        self.gating_network = GatingNetwork(
            input_dim=self.hidden_size, 
            num_experts=gating_num_experts,
            hidden_dim=self.moe_router_hidden_dim, 
            top_k=self.top_k
        )      

        self.head = nn.Linear(self.hidden_size, 2)
        
        self.contrastive_loss_fn = losses.ContrastiveLoss(pos_margin=self.contrastive_pos_margin, neg_margin=self.contrastive_neg_margin)

        self.load_and_replace_from_pretrained(clip_model)

    def load_and_replace_from_pretrained(self, pretrained_model):
        vision_model = pretrained_model.vision_model
        
        self.embeddings.load_state_dict(vision_model.embeddings.state_dict())
        self.ln_pre.load_state_dict(vision_model.pre_layrnorm.state_dict())
        self.ln_post.load_state_dict(vision_model.post_layernorm.state_dict())
        # self.visual_projection.load_state_dict(pretrained_model.visual_projection.state_dict())

        for i, pretrained_layer in enumerate(vision_model.encoder.layers):
            moe_layer = self.encoder_layers[i]
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                self._replace_linear_with_svd_moe(
                    getattr(pretrained_layer.self_attn, proj_name),
                    getattr(moe_layer.self_attn, proj_name)
                )
            moe_layer.layer_norm1.load_state_dict(pretrained_layer.layer_norm1.state_dict())
            moe_layer.mlp.load_state_dict(pretrained_layer.mlp.state_dict())
            moe_layer.layer_norm2.load_state_dict(pretrained_layer.layer_norm2.state_dict())

        # Freeze non-MoE parameters
        for name, param in self.named_parameters():
            if 'experts' not in name and 'gating' not in name and 'head' not in name:
                param.requires_grad = False
                    

    def _replace_linear_with_svd_moe(self, original_module: nn.Linear, moe_module):
        original_weight = original_module.weight.data
        moe_module.weight_original_fnorm.data.copy_(torch.norm(original_weight, p='fro'))
        if original_module.bias is not None:
            moe_module.bias.data.copy_(original_module.bias.data)

        U, S, Vh = torch.linalg.svd(original_weight, full_matrices=False)

        # 1. Initialize the main (shared) weight matrix
        r = moe_module.r_main
        U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
        moe_module.weight_main.data.copy_(U_r @ torch.diag(S_r) @ Vh_r)

        moe_module.U_r.data.copy_(U_r)
        moe_module.V_r.data.copy_(Vh_r)

        # 2. Define the SINGLE, SHARED chunk from ALL remaining singular values.
        chunk_start_rank = r

        # Check if there are any singular values left.
        if chunk_start_rank >= len(S):
            # If not, initialize all experts to zero.
            for i in range(moe_module.num_experts):
                moe_module.U_experts[i].data.zero_()
                moe_module.S_experts[i].data.zero_()
                moe_module.V_experts[i].data.zero_()
            return

        # The chunk is everything that remains.
        U_chunk, S_chunk, Vh_chunk = U[:, chunk_start_rank:], S[chunk_start_rank:], Vh[chunk_start_rank:, :]

        # The expert's parameter dimension is fixed at rank_per_expert.
        # We must pad the chunk if its actual rank is smaller.
        actual_chunk_rank = U_chunk.shape[1]

        if actual_chunk_rank < moe_module.rank_per_expert:
            pad_rank = moe_module.rank_per_expert - actual_chunk_rank
            U_chunk = F.pad(U_chunk, (0, pad_rank))
            S_chunk = F.pad(S_chunk, (0, pad_rank))
            Vh_chunk = F.pad(Vh_chunk, (0, 0, 0, pad_rank))

        # 3. Assign the SAME shared chunk to all experts.
        for i in range(moe_module.num_experts):
            moe_module.U_experts[i].data.copy_(U_chunk)
            moe_module.S_experts[i].data.copy_(S_chunk)
            moe_module.V_experts[i].data.copy_(Vh_chunk)

    def set_training_mode(self, mode: str, active_expert_idx: int = None):
        """
        Sets the training mode of the model, allowing specific parts of the parameters to be frozen/unfrozen.
        This implementation avoids fragile name-based matching and correctly handles all training modes.

        Args:
            mode (str): Training mode. Possible values are:
                - 'hard_sampling': A stage of two-stage training. Freezes everything except for one active expert's parameters.
                - 'router_training': A stage of two-stage training. Freezes all experts and only trains the gating network and head.
                - 'standard': Standard end-to-end MoE training. Trains the gating network, all experts, and the head simultaneously.
            active_expert_idx (int, optional): The index of the expert to be trained in 'hard_sampling' mode.
                                                This parameter is required for 'hard_sampling' mode. Defaults to None.
        """
        if mode not in ['hard_sampling', 'router_training', 'standard']:
            raise ValueError("Training mode must be one of 'hard_sampling', 'router_training', or 'standard'")

        if mode == 'hard_sampling' and (active_expert_idx is None or active_expert_idx < 0):
            raise ValueError("A valid active_expert_idx must be provided for 'hard_sampling' mode.")

        # Store the state in the model instance
        self.training_mode = mode
        self.active_expert_idx = active_expert_idx if mode == 'hard_sampling' else None

        # 1. Start by freezing all parameters by default. This is the safest base state.
        for param in self.parameters():
            param.requires_grad = False

        # 2. Selectively unfreeze parameters based on the chosen mode.
        if mode == 'standard':
            # In standard MoE training, train the router, all experts, and the final head.
            # Unfreeze the gating network (router).
            for param in self.gating_network.parameters():
                param.requires_grad = True
            
            # Unfreeze the classification head.
            for param in self.head.parameters():
                param.requires_grad = True

            # Unfreeze ALL expert parameters.
            for layer in self.encoder_layers:
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                    moe_linear_layer = getattr(layer.self_attn, proj_name)
                    for param in moe_linear_layer.U_experts:
                        param.requires_grad = True
                    for param in moe_linear_layer.S_experts:
                        param.requires_grad = True
                    for param in moe_linear_layer.V_experts:
                        param.requires_grad = True
        
        elif mode == 'router_training':
            # In router_training stage, only train the gating network and the classification head.
            for param in self.gating_network.parameters():
                param.requires_grad = True
            for param in self.head.parameters():
                param.requires_grad = True
        
        elif mode == 'hard_sampling':
            # In hard_sampling stage, unfreeze only the parameters of the single active expert.
            if self.active_expert_idx >= self.num_experts:
                 raise ValueError(f"active_expert_idx ({self.active_expert_idx}) is out of bounds for num_experts ({self.num_experts}).")

            # Iterate through each MoE layer to find and unfreeze the active expert.
            for layer in self.encoder_layers:
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                    moe_linear_layer = getattr(layer.self_attn, proj_name)
                    
                    # Unfreeze the parameters (U, S, V) for the specified active expert by direct indexing.
                    moe_linear_layer.U_experts[self.active_expert_idx].requires_grad = True
                    moe_linear_layer.S_experts[self.active_expert_idx].requires_grad = True
                    moe_linear_layer.V_experts[self.active_expert_idx].requires_grad = True

            for param in self.head.parameters():
                param.requires_grad = True

        print(f"Successfully set training mode to '{self.training_mode}'" + (f" (active expert: {self.active_expert_idx})" if self.training_mode == 'hard_sampling' else ""))
        
        # For verification, print the trainable parameters.
        # print("Currently trainable parameters:")
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"- {name}")


    # def forward(self, images, inference=False) -> dict:
    #     """
    #     Args:
    #         images (torch.Tensor): Input image tensor.
    #         inference (bool, optional): Whether it is inference mode. Defaults to False.
    #     """
    #     batch_size = images.size(0)
    #     hidden_states = self.embeddings(images)

    #     gating_outputs = {}

    #     # Determine how gate_weights are generated based on active_expert_idx
    #     if self.training_mode == 'hard_sampling':
    #         if self.active_expert_idx is None:
    #             raise ValueError("In hard_sampling mode, active_expert_idx must be set.")
            
    #         hard_sampling_target = torch.full((batch_size,), self.active_expert_idx, dtype=torch.long, device=hidden_states.device)
    #         gating_outputs['top_k_indices'] = hard_sampling_target.unsqueeze(1)
    #         gating_outputs['top_k_gates'] = torch.ones_like(gating_outputs['top_k_indices'], dtype=hidden_states.dtype)
    #         gating_outputs['balance_loss'] = torch.tensor(0.0, device=hidden_states.device)
    #     else: # Covers 'router_training' and 'standard'
    #         patch_tokens = hidden_states[:, 1:, :]
    #         routing_features = torch.mean(patch_tokens, dim=1)
    #         gating_outputs = self.gating_network(routing_features)

    #     final_gates = torch.zeros(batch_size, self.num_experts, device=hidden_states.device)
        
    #     final_gates.scatter_(-1, gating_outputs['top_k_indices'], gating_outputs['top_k_gates'])

    #     hidden_states = self.ln_pre(hidden_states)

    #     for layer_module in self.encoder_layers:
    #         hidden_states = layer_module(hidden_states, gating_outputs=gating_outputs)[0]
            
    #     pooled_output = self.ln_post(hidden_states[:, 0, :])
    #     pred = self.head(pooled_output)
    #     prob = torch.softmax(pred, dim=1)[:, 1]

    #     return {
    #         'cls': pred, 
    #         'prob': prob, 
    #         'feat': pooled_output, 
    #         'balance_loss': gating_outputs['balance_loss'],
    #         'final_gates': final_gates
    #     }


    def forward(self, images, inference=False) -> dict:
        """
        Args:
            images (torch.Tensor): Input image tensor.
            inference (bool, optional): Whether it is inference mode. Defaults to False.
        """
        batch_size = images.size(0)
        hidden_states = self.embeddings(images)

        gating_outputs = {}

        # Stage 1 expert training ('hard_sampling'), Determine how gate_weights are generated based on active_expert_idx
        if self.training_mode == 'hard_sampling':
            if self.active_expert_idx is None:
                raise ValueError("In hard_sampling mode, active_expert_idx must be set.")
            
            hard_sampling_target = torch.full((batch_size,), self.active_expert_idx, dtype=torch.long, device=hidden_states.device)
            gating_outputs['top_k_indices'] = hard_sampling_target.unsqueeze(1)
            gating_outputs['top_k_gates'] = torch.ones_like(gating_outputs['top_k_indices'], dtype=hidden_states.dtype)
            gating_outputs['balance_loss'] = torch.tensor(0.0, device=hidden_states.device)
        
        # Stage 2 routing ('router_training') or inference for a Hybrid model
        elif self.is_hybrid:
            # 1. Create a "pass-through" gating output for the fixed artifact expert (the last expert).
            artifact_expert_indices = torch.full(
                (batch_size, 1), 
                self.artifact_expert_idx, # Use the last expert index
                dtype=torch.long, 
                device=hidden_states.device
            )
            artifact_expert_gates = torch.ones(
                (batch_size, 1), 
                dtype=hidden_states.dtype, 
                device=hidden_states.device
            )

            # 2. Get routing decisions for the SEMANTIC experts (experts 0, 1, ..., N-2).
            routing_features = torch.mean(hidden_states[:, 1:, :], dim=1)
            semantic_gating_outputs = self.gating_network(routing_features)
            
            # The indices from the router are already correct (0 to N-2), no shift needed.
            semantic_expert_indices = semantic_gating_outputs['top_k_indices']
            
            # 3. Combine the routed semantic experts and the fixed artifact expert.
            gating_outputs['top_k_indices'] = torch.cat([semantic_expert_indices, artifact_expert_indices], dim=1)
            gating_outputs['top_k_gates'] = torch.cat([semantic_gating_outputs['top_k_gates'], artifact_expert_gates], dim=1)
            
            # The balance loss only comes from the semantic router, which is correct.
            gating_outputs['balance_loss'] = semantic_gating_outputs['balance_loss']
        
        # Standard end-to-end training or inference for a Standard MoE model
        else:
            patch_tokens = hidden_states[:, 1:, :]
            routing_features = torch.mean(patch_tokens, dim=1)
            gating_outputs = self.gating_network(routing_features)            


        final_gates = torch.zeros(batch_size, self.num_experts, device=hidden_states.device)
        
        final_gates.scatter_(-1, gating_outputs['top_k_indices'], gating_outputs['top_k_gates'])

        hidden_states = self.ln_pre(hidden_states)

        for layer_module in self.encoder_layers:
            hidden_states = layer_module(hidden_states, gating_outputs=gating_outputs)[0]
            
        pooled_output = self.ln_post(hidden_states[:, 0, :])
        pred = self.head(pooled_output)
        prob = torch.softmax(pred, dim=1)[:, 0]

        return prob

    def get_losses(self, pred_dict: dict, labels, criterion) -> dict:
        """
        Calculates losses based on the model's internal training mode.
        """
        pred = pred_dict['cls']
        feat = pred_dict['feat'] # Get features for contrastive loss
        classification_loss = criterion(pred, labels)
        
        orth_loss, keepsv_loss, load_balancing_loss = torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device)
        contrastive_loss = torch.tensor(0.0, device=pred.device) # Init contrastive loss


        num_moe_layers = sum(1 for module in self.modules() if isinstance(module, SVDMoeLinear))

        # Orthogonal loss is not calculated during router_training.
        if self.training_mode != 'router_training' and num_moe_layers > 0:
            current_orth_loss = 0.0
            for module in self.modules():
                if isinstance(module, SVDMoeLinear):
                    if self.training_mode == 'hard_sampling':
                        current_orth_loss += module.compute_targeted_orthogonal_loss(self.active_expert_idx)
                    elif self.training_mode == 'standard':
                        current_orth_loss += module.compute_full_orthogonal_loss()
            orth_loss = current_orth_loss / num_moe_layers
        
        # Balance loss is only relevant when the router is active.
        if self.training_mode == 'router_training' or self.training_mode == 'standard':
            load_balancing_loss = pred_dict['balance_loss']
        
        # KeepSV loss is now calculated during 'hard_sampling' AND 'standard'.
        if (self.training_mode == 'hard_sampling' or self.training_mode == 'standard') and num_moe_layers > 0:
            current_keepsv_loss = 0.0
            avg_gate_weights = torch.mean(pred_dict['final_gates'], dim=0)
            for module in self.modules():
                if isinstance(module, SVDMoeLinear):
                    current_keepsv_loss += module.compute_moe_keepsv_loss(avg_gate_weights)
            keepsv_loss = current_keepsv_loss / num_moe_layers

        # Calculate contrastive loss based on DRCT paper [cite: 197, 201]
        # This loss is applied on the output features 'feat' and labels
        contrastive_loss = self.contrastive_loss_fn(feat, labels)

        total_loss = classification_loss + \
                    self.moe_lambda_orth * orth_loss + \
                    self.moe_lambda_balance * load_balancing_loss + \
                    self.moe_lambda_sv * keepsv_loss + \
                    self.moe_lambda_contrastive * contrastive_loss # Add contrastive loss to total

        return {
            'overall_loss': total_loss,
            'classification_loss': classification_loss.detach(),
            'orth_loss': orth_loss.detach(),
            'balance_loss': load_balancing_loss.detach(),
            'keepsv_loss': keepsv_loss.detach(),
            'contrastive_loss': contrastive_loss.detach() # Return for logging
        }




class ViTMoEAttention(nn.Module):
    def __init__(self, config: CLIPVisionConfig, num_experts: int, r_main: int, rank_per_expert: int, artifact_expert_idx: int):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.q_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx)
        self.k_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx)
        self.v_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx)
        self.out_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        gating_outputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Input shape: Batch x Time x Channel
        attention_mask: (batch, 1, a_seq_len, b_seq_len)
        """
        bsz, tgt_len, embed_dim = hidden_states.size()

        # 1. Obtain Q, K, V through SVD-MoE layers and scale Q
        query_states = self.q_proj(hidden_states, gating_outputs) * self.scale
        key_states = self.k_proj(hidden_states, gating_outputs)
        value_states = self.v_proj(hidden_states, gating_outputs)
        
        # 2. Reshape Q, K, V for multi-head attention computation
        # view(bsz, seq_len, num_heads, head_dim) -> transpose(1,2) -> (bsz, num_heads, seq_len, head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)

        # 3. Manually compute attention scores
        # (bsz, num_heads, seq_len, head_dim) @ (bsz, num_heads, head_dim, seq_len) -> (bsz, num_heads, seq_len, seq_len)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = query_states.reshape(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # 4. Apply Attention Mask (if provided)
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            # Mask shape should be (bsz, 1, tgt_len, src_len), will be broadcasted to (bsz, num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # 5. Apply Softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        # 6. (Optional) Save interpretable attention weights
        attn_weights_reshaped = None
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            # Note: To ensure gradients, the official implementation reshapes again, we follow this
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)

        # 7. Apply Attention Dropout
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # 8. Multiply attention probabilities with Value
        attn_output = torch.bmm(attn_probs, value_states)

        # 9. Reshape and apply the final output projection layer
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output, gating_outputs)

        return attn_output, attn_weights_reshaped
    

class ViTMoELayer(nn.Module):
    def __init__(self, config: CLIPVisionConfig, num_experts: int, r_main: int, rank_per_expert: int, artifact_expert_idx: int):
        super().__init__()
        self.self_attn = ViTMoEAttention(config, num_experts, r_main, rank_per_expert, artifact_expert_idx)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, gating_outputs: Dict[str, torch.Tensor], attention_mask: Optional[torch.Tensor] = None):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        # Pass attention_mask to self_attn
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states, 
            gating_outputs=gating_outputs,
            attention_mask=attention_mask
        )

        # hidden_states, attn_weights = checkpoint(
        #     lambda hs, go, am: self.self_attn(hidden_states=hs, gating_outputs=go, attention_mask=am),
        #     hidden_states,
        #     gating_outputs,
        #     attention_mask,
        #     use_reentrant=False
        # )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Maintain the return value format consistent with the official EncoderLayer (tuple)
        return (hidden_states, ) 


class SVDMoeLinear(nn.Module):
    """
    SVD orthogonal subspace linear layer supporting multiple experts and dynamic gating.
    """
    def __init__(self, in_features, out_features, r_main, num_experts, rank_per_expert, artifact_expert_idx, bias=True):
        super(SVDMoeLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r_main = r_main
        self.num_experts = num_experts
        self.rank_per_expert = rank_per_expert
        self.artifact_expert_idx = artifact_expert_idx

        self.register_buffer('weight_main', torch.zeros(out_features, in_features))
        self.register_buffer('U_r', torch.zeros(out_features, r_main))
        self.register_buffer('V_r', torch.zeros(r_main, in_features))
        
        self.U_experts = nn.ParameterList([nn.Parameter(torch.zeros(out_features, rank_per_expert)) for _ in range(num_experts)])
        self.S_experts = nn.ParameterList([nn.Parameter(torch.zeros(rank_per_expert)) for _ in range(num_experts)])
        self.V_experts = nn.ParameterList([nn.Parameter(torch.zeros(rank_per_expert, in_features)) for _ in range(num_experts)])


        self.register_buffer('weight_original_fnorm', torch.tensor(0.0))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)


    def forward(self, x: torch.Tensor, gating_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        An efficient MoE forward pass that computes sparsely.
        """
        # 1. Compute the dense main path, which is applied to all inputs.
        output_main = F.linear(x, self.weight_main, None)
        
        # 2. Get routing decisions from the gating network output.
        top_k_indices = gating_outputs['top_k_indices'] # Shape: [B, k]
        top_k_gates = gating_outputs['top_k_gates']     # Shape: [B, k]
        k = top_k_indices.size(1)

        # 3. Prepare for sparse computation.
        expert_output = torch.zeros_like(output_main)
        # Stack all expert parameters once for efficient indexing.
        U_all = torch.stack([p for p in self.U_experts])
        S_all = torch.stack([p for p in self.S_experts])
        V_all = torch.stack([p for p in self.V_experts])

        # 4. Loop over the k choices. This is highly efficient as k is small.
        for i in range(k):
            # Get the expert indices and corresponding gate weights for the i-th choice.
            chosen_expert_indices = top_k_indices[:, i]  # Shape: [B]
            gate_values = top_k_gates[:, i].unsqueeze(-1) # Shape: [B, 1]

            # Use torch.gather (via simple indexing) to efficiently collect parameters for the chosen experts.
            U_batch = U_all[chosen_expert_indices] # Shape: [B, O, r]
            S_batch = S_all[chosen_expert_indices] # Shape: [B, r]
            V_batch = V_all[chosen_expert_indices] # Shape: [B, r, I]

            # Dynamically construct the residual weight matrices only for the chosen experts.
            W_residual_batch = U_batch @ torch.diag_embed(S_batch) @ V_batch # Shape: [B, O, I]
            
            # Apply the expert weights based on input dimension (3D for sequences, 2D for single vectors).
            if x.dim() == 3: # Case: [Batch, Sequence, Features]
                current_expert_output = torch.einsum('bsi,boi->bso', x, W_residual_batch)
                # Apply the gate weight and accumulate the result.
                expert_output += current_expert_output * gate_values.unsqueeze(-1)
            else: # Case: [Batch, Features]
                current_expert_output = torch.einsum('bi,boi->bo', x, W_residual_batch)
                expert_output += current_expert_output * gate_values
        
        final_output = output_main + expert_output
        if self.bias is not None:
            final_output = final_output + self.bias        
        # 5. Final output is the sum of the main path and the weighted expert path.
        return final_output


    def _calculate_pairwise_loss(self, base1, base2):
        """Helper function to calculate the orthogonal loss between two basis matrices."""
        error = base1.t() @ base2
        return torch.norm(error, p='fro')

    def compute_targeted_orthogonal_loss(self, active_expert_idx: int) -> torch.Tensor:
        """
        Efficient orthogonal loss for the hard sampling stage.
        - If the active expert is the artifact expert, computes loss only against the main space.
        - If the active expert is a semantic expert, computes loss against the main space
          and previously trained *semantic* experts, skipping the artifact expert.
        """

        loss, num_pairs = 0.0, 0
        active_U = F.normalize(self.U_experts[active_expert_idx], dim=0)
        active_V_t = F.normalize(self.V_experts[active_expert_idx], dim=1).t()
        
        # 1. Loss vs. Main space
        loss += self._calculate_pairwise_loss(self.U_r, active_U)
        loss += self._calculate_pairwise_loss(self.V_r.t(), active_V_t)
        num_pairs += 1


        # If the active expert is the artifact expert, we are done.
        if active_expert_idx == self.artifact_expert_idx:
            return loss / num_pairs


        # 2. If it's a semantic expert, also compute loss vs. previously trained semantic experts.
        for i in range(active_expert_idx):
            # Skip comparison with the artifact expert.
            if i == self.artifact_expert_idx:
                continue
            
            prev_U = F.normalize(self.U_experts[i], dim=0)
            prev_V_t = F.normalize(self.V_experts[i], dim=1).t()
            loss += self._calculate_pairwise_loss(prev_U, active_U)
            loss += self._calculate_pairwise_loss(prev_V_t, active_V_t)
            num_pairs += 1
            
        return loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=self.weight_main.device)


    def compute_full_orthogonal_loss(self) -> torch.Tensor:
        """
        Computes the full pairwise orthogonal loss between all components.
        Used for end-to-end finetuning stages.
        """
        all_U_bases = [self.U_r] + [F.normalize(u, dim=0) for u in self.U_experts]
        all_V_bases_t = [self.V_r.t()] + [F.normalize(v, dim=1).t() for v in self.V_experts]

        loss, num_pairs = 0.0, 0
        
        # Iterate over all unique pairs of basis matrices
        for i in range(len(all_U_bases)):
            for j in range(i + 1, len(all_U_bases)):
                loss += self._calculate_pairwise_loss(all_U_bases[i], all_U_bases[j])
                loss += self._calculate_pairwise_loss(all_V_bases_t[i], all_V_bases_t[j])
                num_pairs += 1
                
        return loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=self.weight_main.device)


    def compute_moe_keepsv_loss(self, avg_gate_weights: torch.Tensor) -> torch.Tensor:
        """Calculates the singular value preservation loss to prevent catastrophic forgetting."""
        if self.weight_original_fnorm.item() == 0:
            return torch.tensor(0.0, device=self.weight_main.device)
        
        stacked_experts = torch.stack([
            self.U_experts[i] @ torch.diag(self.S_experts[i]) @ self.V_experts[i] 
            for i in range(self.num_experts)
        ], dim=0)
        
        # Compute the average residual weight based on average gating
        avg_residual_weight = torch.einsum('e,eoi->oi', avg_gate_weights, stacked_experts)
        # Compute the average final weight
        avg_current_weight = self.weight_main + avg_residual_weight
        # Compute the difference between its norm and the original norm
        avg_current_fnorm = torch.norm(avg_current_weight, p='fro')
        loss = torch.abs(avg_current_fnorm**2 - self.weight_original_fnorm**2)
        return loss



class GatingNetwork(nn.Module):
    """
    A sparse gating network that selects the top-k experts for each input token.
    It also calculates a load balancing loss to encourage all experts to be used evenly.
    """
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 256, top_k: int = 2):
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Gating Network's forward pass. Returns a dictionary with routing decisions and losses.
        --- REWRITTEN LOGIC ---
        """
        logits = self.network(x) # Shape: [B, N]
        
        # 1. Select the top k experts and their corresponding scores (logits).
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1) # Shape: [B, k]
        
        # 2. Apply softmax to the scores of the chosen experts to get their weights.
        top_k_gates = F.softmax(top_k_logits, dim=-1) # Shape: [B, k]

        # 3. Calculate the load balancing loss (an auxiliary loss for training stability).
        # This loss provides a differentiable signal to train the router's selection process.
        router_probs = F.softmax(logits, dim=-1) # Probabilities over all experts
        # A sparse mask indicating which experts were chosen
        sparse_mask = torch.zeros_like(logits).scatter_(-1, top_k_indices, 1.0)
        # The frequency of each expert being chosen
        tokens_per_expert = torch.mean(sparse_mask.float(), dim=0)
        # The average probability mass assigned to each expert
        router_prob_per_expert = torch.mean(router_probs, dim=0)
        # The loss is the dot product, encouraging both distributions to be uniform.
        load_balancing_loss = self.num_experts * torch.sum(tokens_per_expert * router_prob_per_expert) 

        # print(logits)

        # Return a dictionary containing all necessary information for the MoE layers and loss calculation.
        return {
            'top_k_indices': top_k_indices,     # The indices of the chosen experts
            'top_k_gates': top_k_gates,         # The weights for the chosen experts
            'balance_loss': load_balancing_loss # The calculated load balancing loss
        }
    
def process_images(image_paths):
    transform_pipeline = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    tensors_list = []

    for image in image_paths:
        # image = Image.open(path).convert('RGB')
        tensor = transform_pipeline(image)
        tensors_list.append(tensor)
    batch_tensor = torch.stack(tensors_list, dim=0)

    return batch_tensor

class EffortScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        moe_config_path = '../RealGen/models/OmniAID/config.json'
        with open(moe_config_path, 'r') as f:
            moe_config = json.load(f)
        moe_config = SimpleNamespace(**moe_config)
        self.model = EffortMoeDetector(config=moe_config)
        checkpoint = torch.load('../RealGen/models/OmniAID/router/checkpoint-0.pth', map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model'], strict=True) 
        self.model.to(self.device)
        self.model.requires_grad_(False)

    
    @torch.no_grad()
    def __call__(self, images):
        
        rewards = []
        for image in tqdm(images):
            tensor_with_batch = process_images([image]).to(self.device)
            reward = self.model(tensor_with_batch).detach().cpu().tolist()
            rewards.append(reward[0])
        return rewards
    


def main():
    scorer = EffortScorer(
        device="cuda",
        dtype=torch.bfloat16
    )

    for model_type in ['sd3_rewrite']:
        image_paths=[
            'xxx'
        ]

        image_paths = [Image.open(img).convert('RGB') for img in image_paths]
        print(f"{model_type}: {scorer(image_paths)}")

if __name__ == "__main__":
    main()