import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Union, List
from transformers import CLIPModel, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings, CLIPMLP
from torch.utils.checkpoint import checkpoint


class OmniAID_LoRA(nn.Module):
    def __init__(self, config=None):
        super(OmniAID_LoRA, self).__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.lora_r_expert = config.rank_per_expert

        self.moe_lambda_orth = config.moe_lambda_orth
        self.moe_lambda_balance = config.moe_lambda_balance
        self.moe_lambda_gating_cls = config.moe_lambda_gating_cls
        self.lora_dropout = config.dropout

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
            # Use the last expert as the artifact expert.
            self.artifact_expert_idx = config.num_experts - 1
            self.num_semantic_experts = config.num_experts - 1
            if self.num_semantic_experts <= 0:
                raise ValueError("num_experts must be at least 2 for the Hybrid MoE model (1 fixed, >=1 semantic).")
            gating_num_experts = self.num_semantic_experts
        else:
            print("Initializing in STANDARD MoE architectural mode.")
            self.artifact_expert_idx = -1  # A value that will never match an expert index
            gating_num_experts = self.num_experts


        pretrained_path = config.CLIP_path

        # Initialize an independent Vision Model as the feature extractor. This model will be used exclusively to generate features for the router.
        self.feature_extractor = CLIPModel.from_pretrained(pretrained_path).vision_model
        self.feature_extractor.eval()

        vision_config = self.feature_extractor.config
        self.hidden_size = vision_config.hidden_size
        
        print(f"LoRA MoE Configuration: Expert Rank={self.lora_r_expert}, Num Experts={self.num_experts}")
        
        # Build the MoE backbone network
        self.embeddings = CLIPVisionEmbeddings(vision_config)
        self.ln_pre = nn.LayerNorm(vision_config.hidden_size)
        self.encoder_layers = nn.ModuleList([
                    ViTMoELayer(vision_config, self.num_experts, self.lora_r_expert, self.artifact_expert_idx, self.lora_dropout) 
                    for _ in range(vision_config.num_hidden_layers)
                ])
        self.ln_post = nn.LayerNorm(self.hidden_size, eps=vision_config.layer_norm_eps)
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
        self.ln_pre.load_state_dict(pretrained_model.pre_layrnorm.state_dict())
        self.ln_post.load_state_dict(pretrained_model.post_layernorm.state_dict())

        for moe_layer, pretrained_layer in zip(self.encoder_layers, pretrained_model.encoder.layers):
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                self._init_lora_moe_weights(
                    getattr(pretrained_layer.self_attn, proj_name),
                    getattr(moe_layer.self_attn, proj_name)
                )
            moe_layer.layer_norm1.load_state_dict(pretrained_layer.layer_norm1.state_dict())
            moe_layer.mlp.load_state_dict(pretrained_layer.mlp.state_dict())
            moe_layer.layer_norm2.load_state_dict(pretrained_layer.layer_norm2.state_dict())

        # Freeze non-MoE parameters
        for name, param in self.named_parameters():
            if 'lora_' not in name and 'gating' not in name and 'head' not in name:
                param.requires_grad = False
                    

    def _init_lora_moe_weights(self, original_module: nn.Linear, moe_module):
        # Copy full original weights to the frozen buffer
        moe_module.weight_fixed.data.copy_(original_module.weight.data)
        if original_module.bias is not None:
            moe_module.bias.data.copy_(original_module.bias.data)

        # Init Expert LoRAs
        for i in range(moe_module.num_experts):
            nn.init.kaiming_uniform_(moe_module.lora_A_experts[i], a=math.sqrt(5))
            nn.init.zeros_(moe_module.lora_B_experts[i])


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

            for layer in self.encoder_layers:
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                    moe_linear_layer = getattr(layer.self_attn, proj_name)
                    for p in moe_linear_layer.lora_A_experts:
                        p.requires_grad = True
                    for p in moe_linear_layer.lora_B_experts:
                        p.requires_grad = True
        
        elif mode == 'router_training':
            print("Unfreezing: Gating Network and Head.")
            for param in self.gating_network.parameters():
                param.requires_grad = True
            for param in self.head.parameters():
                param.requires_grad = True
            
            # Optionally unfreeze specific experts during router training
            if trainable_expert_indices:
                print(f"Additionally unfreezing experts: {trainable_expert_indices}")
                for expert_idx in trainable_expert_indices:
                    if not 0 <= expert_idx < self.num_experts:
                        raise ValueError(f"trainable_expert_index ({expert_idx}) is out of bounds for num_experts ({self.num_experts}).")
                    
                    for layer in self.encoder_layers:
                        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                            moe_linear_layer = getattr(layer.self_attn, proj_name)
                            moe_linear_layer.lora_A_experts[expert_idx].requires_grad = True
                            moe_linear_layer.lora_B_experts[expert_idx].requires_grad = True
        
        elif mode == 'hard_sampling':
            # In hard_sampling stage, unfreeze only the parameters of the single active expert
            if self.active_expert_idx >= self.num_experts:
                 raise ValueError(f"active_expert_idx ({self.active_expert_idx}) is out of bounds for num_experts ({self.num_experts}).")

            for layer in self.encoder_layers:
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                    moe_linear_layer = getattr(layer.self_attn, proj_name)
                    
                    # Unfreeze the parameters (A, B) for the specified active expert by direct indexing
                    moe_linear_layer.lora_A_experts[self.active_expert_idx].requires_grad = True
                    moe_linear_layer.lora_B_experts[self.active_expert_idx].requires_grad = True

            for param in self.head.parameters():
                param.requires_grad = True

        print(f"Successfully set training mode to '{self.training_mode}'" + (f" (active expert: {self.active_expert_idx})" if self.training_mode == 'hard_sampling' else ""))
        
        # print("Currently trainable parameters:")
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"- {name}")

    # For checkpoint
    @staticmethod
    def run_moe_layer(module, hidden_states, gating_outputs):
        return module(hidden_states, gating_outputs=gating_outputs)[0]

    def forward(self, images) -> dict:
        batch_size = images.size(0)

        with torch.no_grad():
            # Use the pooler_output as the routing feature
            routing_features = self.feature_extractor(images, output_hidden_states=False).pooler_output

        hidden_states = self.embeddings(images)
        hidden_states = self.ln_pre(hidden_states)

        gating_outputs = {}

        # Stage 1 expert training ('hard_sampling'). Determine how gate_weights are generated based on active_expert_idx
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

        for layer_module in self.encoder_layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    self.run_moe_layer,
                    layer_module,
                    hidden_states,
                    gating_outputs,
                    use_reentrant=False
                )
            else:
                hidden_states = layer_module(
                    hidden_states, 
                    gating_outputs=gating_outputs, 
                )[0]
            
        pooled_output = self.ln_post(hidden_states[:, 0, :])
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

        num_moe_layers = sum(1 for module in self.modules() if isinstance(module, LoRAMoeLinear))

        # Orthogonal loss is not calculated during router_training
        if self.training_mode != 'router_training' and num_moe_layers > 0:
            current_orth_loss = 0.0
            for module in self.modules():
                if isinstance(module, LoRAMoeLinear):
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


class ViTMoEAttention(nn.Module):
    def __init__(self, config: CLIPVisionConfig, num_experts: int, lora_r_expert: int, artifact_expert_idx: int, lora_dropout: float):
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

        self.q_proj = LoRAMoeLinear(self.embed_dim, self.embed_dim, num_experts, lora_r_expert, artifact_expert_idx, lora_dropout)
        self.k_proj = LoRAMoeLinear(self.embed_dim, self.embed_dim, num_experts, lora_r_expert, artifact_expert_idx, lora_dropout)
        self.v_proj = LoRAMoeLinear(self.embed_dim, self.embed_dim, num_experts, lora_r_expert, artifact_expert_idx, lora_dropout)
        self.out_proj = LoRAMoeLinear(self.embed_dim, self.embed_dim, num_experts, lora_r_expert, artifact_expert_idx, lora_dropout)

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

        query_states = self.q_proj(hidden_states, gating_outputs) * self.scale
        key_states = self.k_proj(hidden_states, gating_outputs)
        value_states = self.v_proj(hidden_states, gating_outputs)
        
        query_states = self._shape(query_states, tgt_len, bsz)
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = query_states.reshape(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        attn_weights_reshaped = None
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output, gating_outputs)

        return attn_output, attn_weights_reshaped
    

class ViTMoELayer(nn.Module):
    def __init__(self, config, num_experts, lora_r_expert, artifact_expert_idx, lora_dropout):
        super().__init__()
        self.self_attn = ViTMoEAttention(config, num_experts, lora_r_expert, artifact_expert_idx, lora_dropout)
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

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Maintain the return value format consistent with the official EncoderLayer (tuple)
        return (hidden_states, ) 


class LoRAMoeLinear(nn.Module):
    def __init__(self, in_features, out_features, num_experts, lora_r_expert, artifact_expert_idx, lora_dropout=0.0, bias=True):
        super(LoRAMoeLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.lora_r_expert = lora_r_expert
        self.artifact_expert_idx = artifact_expert_idx

        # Frozen pretrained weights
        self.register_buffer('weight_fixed', torch.zeros(out_features, in_features))
        
        # Expert LoRA Paths
        self.lora_A_experts = nn.ParameterList([nn.Parameter(torch.zeros(lora_r_expert, in_features)) for _ in range(num_experts)])
        self.lora_B_experts = nn.ParameterList([nn.Parameter(torch.zeros(out_features, lora_r_expert)) for _ in range(num_experts)])

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        if lora_dropout > 0.0:
            self.dropout = nn.Dropout(p=lora_dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor, gating_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Base fixed forward
        output_fixed = F.linear(x, self.weight_fixed, None)
        
        x = self.dropout(x)

        # Expert MoE LoRA path
        top_k_indices = gating_outputs['top_k_indices'] 
        top_k_gates = gating_outputs['top_k_gates']     
        k = top_k_indices.size(1)

        expert_output = torch.zeros_like(output_fixed)
        
        A_all = torch.stack([p for p in self.lora_A_experts]) 
        B_all = torch.stack([p for p in self.lora_B_experts]) 

        original_dim = x.dim()
        if original_dim == 2:
            x = x.unsqueeze(1)
            expert_output = expert_output.unsqueeze(1)

        for i in range(k):
            chosen_expert_indices = top_k_indices[:, i]  
            gate_values = top_k_gates[:, i].unsqueeze(-1).unsqueeze(-1) 

            A_batch = A_all[chosen_expert_indices] 
            B_batch = B_all[chosen_expert_indices] 

            # (x @ A.T) @ B.T
            x_a = torch.bmm(x, A_batch.transpose(1, 2))
            current_expert_out = torch.bmm(x_a, B_batch.transpose(1, 2))
            
            expert_output += current_expert_out * gate_values

        if original_dim == 2:
            expert_output = expert_output.squeeze(1)

        final_output = output_fixed + expert_output
        
        if self.bias is not None:
            final_output = final_output + self.bias
            
        return final_output

    def _calculate_pairwise_orthogonality(self, mat1, mat2, dim_to_normalize):
        v1 = F.normalize(mat1, dim=dim_to_normalize)
        v2 = F.normalize(mat2, dim=dim_to_normalize)
        if dim_to_normalize == 0:
            similarity = v1.t() @ v2
        else:
            similarity = v1 @ v2.t()
        return torch.norm(similarity, p='fro')

    def compute_targeted_orthogonal_loss(self, active_expert_idx: int) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.weight_fixed.device)
        num_pairs = 0
        
        active_A = self.lora_A_experts[active_expert_idx]
        active_B = self.lora_B_experts[active_expert_idx]
        
        # pre experts
        for i in range(active_expert_idx):
            if i == self.artifact_expert_idx: continue
            
            prev_A = self.lora_A_experts[i].detach()
            prev_B = self.lora_B_experts[i].detach()
            
            loss += self._calculate_pairwise_orthogonality(prev_B, active_B, dim_to_normalize=0)
            loss += self._calculate_pairwise_orthogonality(prev_A, active_A, dim_to_normalize=1)
            num_pairs += 1
            
        return loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=self.weight_fixed.device)

    def compute_full_orthogonal_loss(self) -> torch.Tensor:
        all_As = [a for a in self.lora_A_experts]
        all_Bs = [b for b in self.lora_B_experts]

        loss = torch.tensor(0.0, device=self.weight_fixed.device)
        num_pairs = 0
        
        for i in range(len(all_As)):
            for j in range(i + 1, len(all_As)):
                loss += self._calculate_pairwise_orthogonality(all_As[i], all_As[j], dim_to_normalize=1)
                loss += self._calculate_pairwise_orthogonality(all_Bs[i], all_Bs[j], dim_to_normalize=0)
                num_pairs += 1
                
        return loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=self.weight_fixed.device)


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