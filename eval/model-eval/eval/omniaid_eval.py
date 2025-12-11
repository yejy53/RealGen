import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from typing import Optional, Tuple, Dict
from transformers import CLIPModel, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings, CLIPMLP
from PIL import Image
from tqdm import tqdm
import types

class OmniAID(nn.Module):
    def __init__(self, config=None):
        super(OmniAID, self).__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.rank_per_expert = config.rank_per_expert
        self.moe_router_hidden_dim = config.moe_router_hidden_dim
        self.top_k = config.moe_top_k
        
        self.is_hybrid = config.is_hybrid

        if self.is_hybrid:
            self.artifact_expert_idx = config.num_experts - 1
            self.num_semantic_experts = config.num_experts - 1
            gating_num_experts = self.num_semantic_experts
        else:
            self.artifact_expert_idx = -1
            gating_num_experts = self.num_experts

        pretrained_path = config.CLIP_path
        clip_model = CLIPModel.from_pretrained(pretrained_path)
        vision_config = clip_model.vision_model.config
        self.hidden_size = vision_config.hidden_size
        
        total_rank = vision_config.hidden_size
        residual_rank = self.rank_per_expert
        r_main = total_rank - residual_rank
        
        self.embeddings = CLIPVisionEmbeddings(vision_config)
        self.ln_pre = nn.LayerNorm(vision_config.hidden_size)
        self.encoder_layers = nn.ModuleList([
            ViTMoELayer(vision_config, self.num_experts, r_main, self.rank_per_expert, self.artifact_expert_idx) 
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
        
        self.load_and_replace_from_pretrained(clip_model)

    def load_and_replace_from_pretrained(self, pretrained_model):
        vision_model = pretrained_model.vision_model
        
        self.embeddings.load_state_dict(vision_model.embeddings.state_dict())
        self.ln_pre.load_state_dict(vision_model.pre_layrnorm.state_dict())
        self.ln_post.load_state_dict(vision_model.post_layernorm.state_dict())

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

    def _replace_linear_with_svd_moe(self, original_module: nn.Linear, moe_module):

        original_weight = original_module.weight.data
        U, S, Vh = torch.linalg.svd(original_weight, full_matrices=False)
        r = moe_module.r_main
        U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
        moe_module.weight_main.data.copy_(U_r @ torch.diag(S_r) @ Vh_r)
        
        pass 

    def forward(self, images) -> dict:
        batch_size = images.size(0)
        hidden_states = self.embeddings(images)
        routing_features = torch.mean(hidden_states[:, 1:, :], dim=1)

        if self.is_hybrid:
            artifact_expert_indices = torch.full((batch_size, 1), self.artifact_expert_idx, dtype=torch.long, device=hidden_states.device)
            artifact_expert_gates = torch.ones((batch_size, 1), dtype=hidden_states.dtype, device=hidden_states.device)

            semantic_gating_outputs = self.gating_network(routing_features)
            
            gating_outputs = {
                'top_k_indices': torch.cat([semantic_gating_outputs['top_k_indices'], artifact_expert_indices], dim=1),
                'top_k_gates': torch.cat([semantic_gating_outputs['top_k_gates'], artifact_expert_gates], dim=1)
            }
        
        else:
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


class ViTMoEAttention(nn.Module):
    def __init__(self, config: CLIPVisionConfig, num_experts: int, r_main: int, rank_per_expert: int, artifact_expert_idx: int):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.q_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx)
        self.k_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx)
        self.v_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx)
        self.out_proj = SVDMoeLinear(self.embed_dim, self.embed_dim, r_main, num_experts, rank_per_expert, artifact_expert_idx)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, gating_outputs, attention_mask=None):
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
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.bmm(attn_weights, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output, gating_outputs)
        return attn_output, None

class ViTMoELayer(nn.Module):
    def __init__(self, config: CLIPVisionConfig, num_experts: int, r_main: int, rank_per_expert: int, artifact_expert_idx: int):
        super().__init__()
        self.self_attn = ViTMoEAttention(config, num_experts, r_main, rank_per_expert, artifact_expert_idx)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, gating_outputs, attention_mask=None):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states, gating_outputs=gating_outputs, attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states, )

class SVDMoeLinear(nn.Module):
    def __init__(self, in_features, out_features, r_main, num_experts, rank_per_expert, artifact_expert_idx, bias=True):
        super(SVDMoeLinear, self).__init__()
        self.r_main = r_main
        self.num_experts = num_experts
        self.rank_per_expert = rank_per_expert
        self.register_buffer('weight_main', torch.zeros(out_features, in_features))
        self.U_experts = nn.ParameterList([nn.Parameter(torch.zeros(out_features, rank_per_expert)) for _ in range(num_experts)])
        self.S_experts = nn.ParameterList([nn.Parameter(torch.zeros(rank_per_expert)) for _ in range(num_experts)])
        self.V_experts = nn.ParameterList([nn.Parameter(torch.zeros(rank_per_expert, in_features)) for _ in range(num_experts)])
        self.register_buffer('weight_original_fnorm', torch.tensor(0.0))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor, gating_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        output_main = F.linear(x, self.weight_main, None)
        top_k_indices = gating_outputs['top_k_indices']
        top_k_gates = gating_outputs['top_k_gates']
        k = top_k_indices.size(1)

        expert_output = torch.zeros_like(output_main)
        U_all = torch.stack([p for p in self.U_experts])
        S_all = torch.stack([p for p in self.S_experts])
        V_all = torch.stack([p for p in self.V_experts])

        original_dim = x.dim()
        if original_dim == 2:
            x = x.unsqueeze(1)
            expert_output = expert_output.unsqueeze(1)

        for i in range(k):
            chosen_expert_indices = top_k_indices[:, i]
            gate_values = top_k_gates[:, i].unsqueeze(-1)
            U_batch = U_all[chosen_expert_indices]
            S_batch = S_all[chosen_expert_indices]
            V_batch = V_all[chosen_expert_indices]

            x_v = torch.bmm(x, V_batch.transpose(1, 2))
            x_v_s = x_v * S_batch.unsqueeze(1)
            current_expert_output = torch.bmm(x_v_s, U_batch.transpose(1, 2))
            expert_output += current_expert_output * gate_values.unsqueeze(-1)

        if original_dim == 2:
            expert_output = expert_output.squeeze(1)

        final_output = output_main + expert_output
        if self.bias is not None:
            final_output = final_output + self.bias        
        return final_output

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
        logits = self.network(x)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        return {'top_k_indices': top_k_indices, 'top_k_gates': top_k_gates}


def get_config():
    config = types.SimpleNamespace()
    config.ckpt_path = "../RealGen/models/router/checkpoint-0.pth"  # Path to the checkpoint
    # config.CLIP_path = "openai/clip-vit-large-patch14-336" # Default
    config.CLIP_path = "../RealGen/models/model_path"
    config.num_experts = 6
    config.rank_per_expert = 1
    config.moe_top_k = 2
    config.moe_router_hidden_dim = 256
    config.image_resolution = 336 
    config.is_hybrid = True
    return config

    
def process_images(image_paths):
    transform_pipeline = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    tensors_list = []

    for image in image_paths:
        tensor = transform_pipeline(image)
        tensors_list.append(tensor)
    batch_tensor = torch.stack(tensors_list, dim=0)

    return batch_tensor

class OmniAIDScorer(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        config = get_config()
        self.model = OmniAID(config=config)
        checkpoint = torch.load(config.ckpt_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model'], strict=False) 
        self.model.to(self.device)
        self.model.eval()
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
    scorer = OmniAIDScorer(device="cuda")

    is_rewrite = "short"
    img_list = ['xxx']
    for img_path in img_list:
        image_paths=[]
        for i in range(1041):
            path = f"/model-eval/{is_rewrite}-img/{img_path}/photo/{i:05}.jpg"
            if os.path.exists(path):
                image_paths.append(path)
        image_paths = [Image.open(img).convert('RGB') for img in image_paths]
        scores = scorer(image_paths)
        rate = new_list = [1 if number > 0.5 else 0 for number in scores]
        print(f"=========={img_path}==========")
        print(sum(scores)/len(scores))
        print(sum(rate)/len(rate))


if __name__ == "__main__":
    main()




    
