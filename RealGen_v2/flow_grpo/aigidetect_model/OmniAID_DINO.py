import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union, List
from transformers import DINOv3ViTModel, DINOv3ViTConfig
from transformers.models.dinov3_vit.modular_dinov3_vit import (
    DINOv3ViTEmbeddings, 
    DINOv3ViTRopePositionEmbedding, 
    DINOv3ViTMLP, 
    DINOv3ViTGatedMLP, 
    DINOv3ViTLayerScale, 
    DINOv3ViTDropPath,
    apply_rotary_pos_emb
)
from torch.utils.checkpoint import checkpoint


class OmniAID_DINO(nn.Module):
    def __init__(self, config=None):
        super(OmniAID_DINO, self).__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.rank_per_expert = config.rank_per_expert

        self.moe_lambda_orth = config.moe_lambda_orth
        self.moe_lambda_balance = config.moe_lambda_balance
        self.moe_lambda_gating_cls = config.moe_lambda_gating_cls
        self.svd_dropout = config.dropout

        self.moe_router_hidden_dim = config.moe_router_hidden_dim
        self.top_k = config.moe_top_k

        self.gradient_checkpointing = config.gradient_checkpointing_enable

        self.gating_criterion = torch.nn.CrossEntropyLoss()

        self.training_mode = "standard" # Default runtime mode, can be changed by set_training_mode
        self.active_expert_idx = None

        # This property is fixed after initialization
        self.is_hybrid = config.is_hybrid

        if self.is_hybrid:
            print("Initializing in HYBRID MoE architectural mode.")
            # Use the last expert as the artifact expert
            self.artifact_expert_idx = config.num_experts - 1
            self.num_semantic_experts = config.num_experts - 1
            if self.num_semantic_experts <= 0:
                raise ValueError("num_experts must be at least 2 for the Hybrid MoE model (1 fixed, >=1 semantic).")
            gating_num_experts = self.num_semantic_experts
        else:
            print("Initializing in STANDARD MoE architectural mode.")
            self.artifact_expert_idx = -1  # A value that will never match an expert index
            gating_num_experts = self.num_experts

        pretrained_path = config.DINOV3_path

        self.feature_extractor = DINOv3ViTModel.from_pretrained(pretrained_path)
        self.feature_extractor.eval()

        vision_config = self.feature_extractor.config
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
        self.embeddings = DINOv3ViTEmbeddings(vision_config)
        self.rope_embeddings = DINOv3ViTRopePositionEmbedding(vision_config)
        self.layer = nn.ModuleList([
            DINOv3ViTMoELayer(vision_config, self.num_experts, r_main, self.rank_per_expert, self.artifact_expert_idx, self.svd_dropout) 
            for _ in range(vision_config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(vision_config.hidden_size, eps=vision_config.layer_norm_eps)
        self.gating_network = GatingNetwork(
            input_dim=self.hidden_size, 
            num_experts=gating_num_experts,
            hidden_dim=self.moe_router_hidden_dim, 
            top_k=self.top_k
        )      
        self.head = nn.Linear(self.hidden_size, 2)
        
        self.load_and_replace_from_pretrained(self.feature_extractor)

    def load_and_replace_from_pretrained(self, pretrained_model):
        self.embeddings.load_state_dict(pretrained_model.embeddings.state_dict())
        self.rope_embeddings.load_state_dict(pretrained_model.rope_embeddings.state_dict())
        self.norm.load_state_dict(pretrained_model.norm.state_dict())

        for moe_layer, pretrained_layer in zip(self.layer, pretrained_model.layer):

            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                self._replace_linear_with_svd_moe(
                    getattr(pretrained_layer.attention, proj_name),
                    getattr(moe_layer.attention, proj_name)
                )            

            moe_layer.mlp.load_state_dict(pretrained_layer.mlp.state_dict())

            moe_layer.norm1.load_state_dict(pretrained_layer.norm1.state_dict())
            moe_layer.norm2.load_state_dict(pretrained_layer.norm2.state_dict())
            moe_layer.layer_scale1.load_state_dict(pretrained_layer.layer_scale1.state_dict())
            moe_layer.layer_scale2.load_state_dict(pretrained_layer.layer_scale2.state_dict())
        
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

        # Initialize the main (shared) weight matrix
        r = moe_module.r_main
        U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
        moe_module.weight_main.data.copy_(U_r @ torch.diag(S_r) @ Vh_r)

        moe_module.U_r.data.copy_(U_r)
        moe_module.V_r.data.copy_(Vh_r)

        # Define the SINGLE, SHARED chunk from ALL remaining singular values
        chunk_start_rank = r

        # Check if there are any singular values left
        if chunk_start_rank >= len(S):
            # If not, initialize all experts to zero
            for i in range(moe_module.num_experts):
                moe_module.U_experts[i].data.zero_()
                moe_module.S_experts[i].data.zero_()
                moe_module.V_experts[i].data.zero_()
            return

        U_chunk, S_chunk, Vh_chunk = U[:, chunk_start_rank:], S[chunk_start_rank:], Vh[chunk_start_rank:, :]

        # The expert's parameter dimension is fixed at rank_per_expert. We must pad the chunk if its actual rank is smaller
        actual_chunk_rank = U_chunk.shape[1]

        if actual_chunk_rank < moe_module.rank_per_expert:
            pad_rank = moe_module.rank_per_expert - actual_chunk_rank
            U_chunk = F.pad(U_chunk, (0, pad_rank))
            S_chunk = F.pad(S_chunk, (0, pad_rank))
            Vh_chunk = F.pad(Vh_chunk, (0, 0, 0, pad_rank))

        # Assign the SAME shared chunk to all experts
        for i in range(moe_module.num_experts):
            moe_module.U_experts[i].data.copy_(U_chunk)
            moe_module.S_experts[i].data.copy_(S_chunk)
            moe_module.V_experts[i].data.copy_(Vh_chunk)

    def set_training_mode(self, mode: str, active_expert_idx: int = None, trainable_expert_indices: list = None):
        """
        Sets the training mode of the model, allowing specific parts of the parameters to be frozen/unfrozen.

        Args:
            mode (str): Training mode. Possible values are:
                - 'hard_sampling': Stage 1. Unfreezes one active expert and the head.
                - 'router_training': Stage 2. Unfreezes the router and the head. Optionally unfreezes specified experts.
                - 'standard': End-to-end. Unfreezes the router, head, and all experts.
            active_expert_idx (int, optional): The index of the expert to be trained in 'hard_sampling' mode. Required for this mode.
            trainable_expert_indices (list, optional): A list of expert indices to also unfreeze during 'router_training' mode.
                                                        Allows for joint training of the router and specific experts.
        """
        if mode not in ['hard_sampling', 'router_training', 'standard']:
            raise ValueError("Training mode must be one of 'hard_sampling', 'router_training', or 'standard'")

        if mode == 'hard_sampling' and (active_expert_idx is None or not isinstance(active_expert_idx, int) or active_expert_idx < 0):
            raise ValueError("A valid integer active_expert_idx must be provided for 'hard_sampling' mode.")
            
        if mode == 'router_training' and trainable_expert_indices is not None and not isinstance(trainable_expert_indices, list):
            raise ValueError("'trainable_expert_indices' must be a list of integers.")


        # Store the state in the model instance
        self.training_mode = mode
        self.active_expert_idx = active_expert_idx if mode == 'hard_sampling' else None

        for param in self.parameters():
            param.requires_grad = False

        if mode == 'standard':
            # In standard MoE training, train the router, all experts, and the final head
            for param in self.gating_network.parameters():
                param.requires_grad = True

            for param in self.head.parameters():
                param.requires_grad = True

            for layer in self.layer:
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    moe_linear_layer = getattr(layer.attention, proj_name)
                    for param in moe_linear_layer.U_experts:
                        param.requires_grad = True
                    for param in moe_linear_layer.S_experts:
                        param.requires_grad = True
                    for param in moe_linear_layer.V_experts:
                        param.requires_grad = True
        
        elif mode == 'router_training':
            print("Unfreezing: Gating Network and Head.")
            for param in self.gating_network.parameters():
                param.requires_grad = True
            for param in self.head.parameters():
                param.requires_grad = True
            
            # Optionally unfreeze specific experts during router training
            if trainable_expert_indices:
                print(f"Additionally unfreezing experts with indices: {trainable_expert_indices}")
                for expert_idx in trainable_expert_indices:
                    if not 0 <= expert_idx < self.num_experts:
                        raise ValueError(f"trainable_expert_index ({expert_idx}) is out of bounds for num_experts ({self.num_experts}).")
                    
                    for layer in self.layer:
                        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                            moe_linear_layer = getattr(layer.attention, proj_name)
                            moe_linear_layer.U_experts[expert_idx].requires_grad = True
                            moe_linear_layer.S_experts[expert_idx].requires_grad = True
                            moe_linear_layer.V_experts[expert_idx].requires_grad = True
        
        elif mode == 'hard_sampling':
            # In hard_sampling stage, unfreeze only the parameters of the single active expert
            if self.active_expert_idx >= self.num_experts:
                 raise ValueError(f"active_expert_idx ({self.active_expert_idx}) is out of bounds for num_experts ({self.num_experts}).")

            for layer in self.layer:
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    moe_linear_layer = getattr(layer.attention, proj_name)
                    
                    # Unfreeze the parameters (U, S, V) for the specified active expert by direct indexing
                    moe_linear_layer.U_experts[self.active_expert_idx].requires_grad = True
                    moe_linear_layer.S_experts[self.active_expert_idx].requires_grad = True
                    moe_linear_layer.V_experts[self.active_expert_idx].requires_grad = True

            for param in self.head.parameters():
                param.requires_grad = True

        print(f"Successfully set training mode to '{self.training_mode}'" + (f" (active expert: {self.active_expert_idx})" if self.training_mode == 'hard_sampling' else ""))
        
        # print("Currently trainable parameters:")
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"- {name}")

    # For checkpoint
    @staticmethod
    def run_moe_layer(module, hidden_states, gating_outputs, position_embeddings):
        return module(hidden_states, gating_outputs=gating_outputs, position_embeddings=position_embeddings)

    def forward(self, images) -> dict:
        images = images.to(self.embeddings.patch_embeddings.weight.dtype)
        batch_size = images.size(0)

        with torch.no_grad():
            # Use the pooler_output as the routing feature
            routing_features = self.feature_extractor(images).pooler_output

        hidden_states = self.embeddings(images)
        position_embeddings = self.rope_embeddings(images)

        gating_outputs = {}

        # Stage 1 expert training ('hard_sampling')
        if self.training_mode == 'hard_sampling':
            if self.active_expert_idx is None:
                raise ValueError("In hard_sampling mode, active_expert_idx must be set.")
            
            hard_sampling_target = torch.full((batch_size,), self.active_expert_idx, dtype=torch.long, device=hidden_states.device)
            gating_outputs['top_k_indices'] = hard_sampling_target.unsqueeze(1)
            gating_outputs['top_k_gates'] = torch.ones_like(gating_outputs['top_k_indices'], dtype=hidden_states.dtype)
            gating_outputs['balance_loss'] = torch.tensor(0.0, device=hidden_states.device)
            gating_outputs['gating_logits'] = None
        
        # Stage 2 routing ('router_training') or inference for a Hybrid model
        elif self.is_hybrid:
            # Create a "pass-through" gating output for the fixed artifact expert (the last expert)
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

            # Get routing decisions for the SEMANTIC experts (experts 0, 1, ..., N-2)
            semantic_gating_outputs = self.gating_network(routing_features)
            
            semantic_expert_indices = semantic_gating_outputs['top_k_indices']
            
            # Combine the routed semantic experts and the fixed artifact expert
            gating_outputs['top_k_indices'] = torch.cat([semantic_expert_indices, artifact_expert_indices], dim=1)
            gating_outputs['top_k_gates'] = torch.cat([semantic_gating_outputs['top_k_gates'], artifact_expert_gates], dim=1)
            
            gating_outputs['balance_loss'] = semantic_gating_outputs['balance_loss']
            gating_outputs['gating_logits'] = semantic_gating_outputs['gating_logits']
        
        # Standard end-to-end training or inference for a Standard MoE model
        else:
            gating_outputs = self.gating_network(routing_features)            

        final_gates = torch.zeros(batch_size, self.num_experts, device=hidden_states.device, dtype=gating_outputs['top_k_gates'].dtype)
        
        final_gates.scatter_(-1, gating_outputs['top_k_indices'], gating_outputs['top_k_gates'])

        for layer_module in self.layer:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    self.run_moe_layer,
                    layer_module,
                    hidden_states,
                    gating_outputs,
                    position_embeddings,
                    use_reentrant=False
                )
            else:
                hidden_states = layer_module(
                    hidden_states, 
                    gating_outputs=gating_outputs, 
                    position_embeddings=position_embeddings
                )
        
        sequence_output = self.norm(hidden_states)    
        pooled_output = sequence_output[:, 0, :]

        pred = self.head(pooled_output)
        prob = torch.softmax(pred, dim=1)[:, 1]

        return {
            'cls': pred,
            'prob': prob,
            'balance_loss': gating_outputs['balance_loss'],
            'gating_logits': gating_outputs['gating_logits'],
            'final_gates': final_gates
        }

    def get_losses(self, pred_dict: dict, labels, expert_domain_labels, criterion) -> dict:
        pred = pred_dict['cls']
        classification_loss = criterion(pred, labels)
        
        orth_loss, load_balancing_loss = torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device)
        gating_classification_loss = torch.tensor(0.0, device=pred.device)

        num_moe_layers = sum(1 for module in self.modules() if isinstance(module, SVDMoeLinear))

        # Orthogonal loss is not calculated during router_training
        if self.training_mode != 'router_training' and num_moe_layers > 0:
            current_orth_loss = 0.0
            for module in self.modules():
                if isinstance(module, SVDMoeLinear):
                    if self.training_mode == 'hard_sampling':
                        current_orth_loss += module.compute_targeted_orthogonal_loss(self.active_expert_idx)
                    elif self.training_mode == 'standard':
                        current_orth_loss += module.compute_full_orthogonal_loss()
            orth_loss = current_orth_loss / num_moe_layers

        # Balance loss is only relevant when the router is active
        if self.training_mode == 'router_training' or self.training_mode == 'standard':
            load_balancing_loss = pred_dict['balance_loss']

        if self.training_mode == 'router_training' or self.training_mode == 'standard':
            gating_logits = pred_dict['gating_logits']
            
            if gating_logits is not None and expert_domain_labels is not None:
                
                valid_mask = (expert_domain_labels >= 0) & (expert_domain_labels < self.gating_network.num_experts)
                
                filtered_logits = gating_logits[valid_mask]
                filtered_labels = expert_domain_labels[valid_mask]

                if filtered_logits.shape[0] > 0:
                    gating_classification_loss = self.gating_criterion(filtered_logits, filtered_labels)


        total_loss = classification_loss + \
                    self.moe_lambda_orth * orth_loss + \
                    self.moe_lambda_balance * load_balancing_loss + \
                    self.moe_lambda_gating_cls * gating_classification_loss

        return {
            'overall_loss': total_loss,
            'classification_loss': classification_loss.detach(),
            'orth_loss': orth_loss.detach(),
            'balance_loss': load_balancing_loss.detach(),
            'gating_cls_loss': gating_classification_loss.detach(),
        }


class DINOv3ViTMoEAttention(nn.Module):
    def __init__(self, config, num_experts, r_main, rank_per_expert, artifact_expert_idx, svd_dropout):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        
        self.q_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx, svd_dropout, config.query_bias)
        self.k_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx, svd_dropout, config.key_bias)
        self.v_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx, svd_dropout, config.value_bias)
        self.o_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx, svd_dropout, config.proj_bias)

    def forward(self, hidden_states, gating_outputs, attention_mask=None, position_embeddings=None):
        batch_size, patches, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states, gating_outputs)
        key_states = self.k_proj(hidden_states, gating_outputs)
        value_states = self.v_proj(hidden_states, gating_outputs)

        query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(batch_size, patches, -1).contiguous()
        attn_output = self.o_proj(attn_output, gating_outputs)

        return attn_output, attn_weights


class DINOv3ViTMoELayer(nn.Module):
    def __init__(self, config: DINOv3ViTConfig, num_experts: int, r_main: int, rank_per_expert: int, artifact_expert_idx: int, svd_dropout: float):
        super().__init__()

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = DINOv3ViTMoEAttention(config, num_experts, r_main, rank_per_expert, artifact_expert_idx, svd_dropout)
        self.layer_scale1 = DINOv3ViTLayerScale(config)
        self.drop_path = DINOv3ViTDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.use_gated_mlp:
            self.mlp = DINOv3ViTGatedMLP(config)
        else:
            self.mlp = DINOv3ViTMLP(config)
            
        self.layer_scale2 = DINOv3ViTLayerScale(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_outputs: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Attention with residual connection
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _ = self.attention(
            hidden_states, 
            gating_outputs, 
            attention_mask=attention_mask, 
            position_embeddings=position_embeddings
        )
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        return hidden_states


class SVDMoeLinear(nn.Module):
    """
    SVD orthogonal subspace linear layer supporting multiple experts and dynamic gating.
    """
    def __init__(self, in_features, out_features, r_main, num_experts, rank_per_expert, artifact_expert_idx, svd_dropout=0.0, bias=True):
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

        if svd_dropout > 0.0:
            self.dropout = nn.Dropout(p=svd_dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor, gating_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        output_main = F.linear(x, self.weight_main, None)

        x = self.dropout(x)
        
        top_k_indices = gating_outputs['top_k_indices'] # Shape: [B, k]
        top_k_gates = gating_outputs['top_k_gates']     # Shape: [B, k]
        k = top_k_indices.size(1)

        expert_output = torch.zeros_like(output_main)
        U_all = torch.stack([p for p in self.U_experts])
        S_all = torch.stack([p for p in self.S_experts])
        V_all = torch.stack([p for p in self.V_experts])

        # Record the original input dimension (2 or 3)
        original_dim = x.dim()
        
        # We use torch.bmm which requires 3D inputs
        # If x is 2D (Batch, In_features), we unsqueeze it to 3D (Batch, 1, In_features)
        if original_dim == 2:
            x = x.unsqueeze(1)  # Shape: [B, I] -> [B, 1, I]
            # We do the same for expert_output to match dimensions for accumulation
            expert_output = expert_output.unsqueeze(1) # Shape: [B, O] -> [B, 1, O]

        # Loop over the k choices
        for i in range(k):
            chosen_expert_indices = top_k_indices[:, i]  # Shape: [B]
            gate_values = top_k_gates[:, i].unsqueeze(-1) # Shape: [B, 1]

            U_batch = U_all[chosen_expert_indices] # Shape: [B, O, r]
            S_batch = S_all[chosen_expert_indices] # Shape: [B, r]
            V_batch = V_all[chosen_expert_indices] # Shape: [B, r, I]

            # 1. (x @ V.T)
            #    Shape: [B, S, I] @ [B, I, r] -> [B, S, r]
            x_v = torch.bmm(x, V_batch.transpose(1, 2))
            
            # 2. (x @ V.T) * S
            #    This is equivalent to (x @ V.T) @ diag(S)
            #    Shape: [B, S, r] * [B, 1, r] (broadcast) -> [B, S, r]
            x_v_s = x_v * S_batch.unsqueeze(1)
            
            # 3. ((x @ V.T) * S) @ U.T is equivalent to `output = x @ (U @ diag(S) @ V).T`
            #    Shape: [B, S, r] @ [B, r, O] -> [B, S, O]
            current_expert_output = torch.bmm(x_v_s, U_batch.transpose(1, 2))
            
            # 4. Apply gate weights
            #    Shape: [B, S, O] * [B, 1, 1] (broadcast) -> [B, S, O]
            expert_output += current_expert_output * gate_values.unsqueeze(-1)

        # If the input was 2D, squeeze the sequence dimension (S=1) back out
        if original_dim == 2:
            expert_output = expert_output.squeeze(1) # Shape: [B, 1, O] -> [B, O]

        final_output = output_main + expert_output
        if self.bias is not None:
            final_output = final_output + self.bias        
        return final_output

    def _calculate_pairwise_loss(self, base1, base2):
        error = base1.t() @ base2
        return torch.norm(error, p='fro')

    def compute_targeted_orthogonal_loss(self, active_expert_idx: int) -> torch.Tensor:
        """
        Efficient orthogonal loss for the hard sampling stage.
        - If the active expert is the artifact expert, computes loss only against the main space.
        - If the active expert is a semantic expert, computes loss against the main space
          and previously trained *semantic* experts, skipping the artifact expert.
        """
        loss = torch.tensor(0.0, device=self.weight_main.device)
        num_pairs = 0
        
        active_U = F.normalize(self.U_experts[active_expert_idx], dim=0)
        active_V_t = F.normalize(self.V_experts[active_expert_idx], dim=1).t()
        
        # Loss vs. Main space
        loss += self._calculate_pairwise_loss(self.U_r, active_U)
        loss += self._calculate_pairwise_loss(self.V_r.t(), active_V_t)
        num_pairs += 1

        if active_expert_idx == self.artifact_expert_idx:
            return loss / num_pairs

        # Loss vs. previously trained semantic experts
        for i in range(active_expert_idx):
            # Skip comparison with the artifact expert
            if i == self.artifact_expert_idx:
                continue
            
            prev_U = F.normalize(self.U_experts[i].detach(), dim=0)
            prev_V_t = F.normalize(self.V_experts[i].detach(), dim=1).t()
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

        loss = torch.tensor(0.0, device=self.weight_main.device)
        num_pairs = 0
        
        # Iterate over all unique pairs of basis matrices
        for i in range(len(all_U_bases)):
            for j in range(i + 1, len(all_U_bases)):
                loss += self._calculate_pairwise_loss(all_U_bases[i], all_U_bases[j])
                loss += self._calculate_pairwise_loss(all_V_bases_t[i], all_V_bases_t[j])
                num_pairs += 1
                
        return loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=self.weight_main.device)


class GatingNetwork(nn.Module):
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
        logits = self.network(x) # Shape: [B, N]
        
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1) # Shape: [B, k]
        top_k_gates = F.softmax(top_k_logits, dim=-1) # Shape: [B, k]

        router_probs = F.softmax(logits, dim=-1)
        sparse_mask = torch.zeros_like(logits).scatter_(-1, top_k_indices, 1.0)
        tokens_per_expert = torch.mean(sparse_mask.float(), dim=0)
        router_prob_per_expert = torch.mean(router_probs, dim=0)
        load_balancing_loss = self.num_experts * torch.sum(tokens_per_expert * router_prob_per_expert) 

        return {
            'gating_logits': logits,
            'top_k_indices': top_k_indices,     # The indices of the chosen experts
            'top_k_gates': top_k_gates,         # The weights for the chosen experts
            'balance_loss': load_balancing_loss
        }