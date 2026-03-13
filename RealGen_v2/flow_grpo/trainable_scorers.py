import math
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP

from flow_grpo.aigidetect_model.OmniAID import OmniAID
from flow_grpo.aigidetect_model.OmniAID_LoRA import OmniAID_LoRA
from flow_grpo.aigidetect_model.OmniAID_DINO import OmniAID_DINO
from flow_grpo.aigidetect_model.OmniAID_DINO_LoRA import OmniAID_DINO_LoRA
from flow_grpo.ema import EMAModuleWrapper

from threading import Lock
from accelerate.logging import get_logger

logger = get_logger(__name__)

class TrainableScorerConfig:
    def __init__(self, model_name):
        self.model_name = model_name
        
        self.CLIP_path = "openai/clip-vit-large-patch14-336"
        self.DINOV3_path = "facebook/dinov3-vitl16-pretrain-lvd1689m"

        # --- OmniAID Series General Config ---
        self.num_experts = 6    
        self.rank_per_expert = 1 if 'dino' in model_name else 8
        self.moe_lambda_orth = 0.001
        self.moe_lambda_balance = 0.0
        self.moe_lambda_gating_cls = 0.1
        self.moe_router_hidden_dim = 256
        self.moe_top_k = 2
        self.dropout = 0.1
        self.is_hybrid = True
        self.gradient_checkpointing_enable = True

        self.mean = [0.485, 0.456, 0.406] if 'dino' in model_name else [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.229, 0.224, 0.225] if 'dino' in model_name else [0.26862954, 0.26130258, 0.27577711]
        self.resolution = 448 if 'dino' in model_name else 336

        if model_name == 'omniaid':
            self.ckpt_path = " "
        elif model_name == 'omniaid-dino':
            self.ckpt_path = " "
        elif model_name == 'omniaid-lora':
            self.ckpt_path = " "
        elif model_name == 'omniaid-dino-lora':
            self.ckpt_path = " "
        else:
            raise ValueError(f"Unknown model architecture for: {model_name}")
            
    def to_dict(self):
        return self.__dict__

#  Unified Scorer Wrapper
class TrainableScorer(nn.Module):
    def __init__(self, model_name, device, dtype=torch.float32, ema_decay=0.99):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        # Load Config directly based on name
        logger.info(f"[TrainableScorer] Initializing configuration for: {model_name}")
        self.config = TrainableScorerConfig(model_name)
        
        # Instantiate Model
        if model_name == 'omniaid':
            logger.info(f"[TrainableScorer] Loading OmniAID architecture...")
            self.model = OmniAID(self.config)
            trainable_keywords = ['head', 'experts'] 
        elif model_name == 'omniaid-lora':
            logger.info(f"[TrainableScorer] Loading OmniAID-LoRA architecture...")
            self.model = OmniAID_LoRA(self.config)
            trainable_keywords = ['head']
        elif model_name == 'omniaid-dino':
            logger.info(f"[TrainableScorer] Loading OmniAID-DINO architecture...")
            self.model = OmniAID_DINO(self.config)
            trainable_keywords = ['head', 'experts']        
        elif model_name == 'omniaid-dino-lora':
            logger.info(f"[TrainableScorer] Loading OmniAID-DINO-LoRA architecture...")
            self.model = OmniAID_DINO_LoRA(self.config)
            trainable_keywords = ['head'] 
        else:
            raise ValueError(f"Unknown model architecture for: {model_name}")

        checkpoint = torch.load(self.config.ckpt_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model'], strict=True) 

        self.model.to(device)
        self.model.to(dtype)

        self.trainable_params = []

        frozen_param_count = 0
        trainable_param_count = 0
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for kw in trainable_keywords:
                if kw in name:
                    param.requires_grad = True
                    self.trainable_params.append(param)
                    break
            
            if param.requires_grad: 
                trainable_param_count += param.numel()
            else: 
                frozen_param_count += param.numel()
            
        logger.info(f"[TrainableScorer] Model Ready.")
        logger.info(f"  > Trainable Params : {trainable_param_count:,} ({trainable_param_count/1e6:.6f}M)")
        logger.info(f"  > Frozen Params    : {frozen_param_count:,} ({frozen_param_count/1e6:.2f}M)")

        self.ema_wrapper = EMAModuleWrapper(
            self.trainable_params, 
            decay=ema_decay, 
            device=device
        )
        self.global_step_d = 0

        self.train_transform = transforms.Compose([
            transforms.Resize([512, 512]),
            transforms.Resize([self.config.resolution, self.config.resolution]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                    [
                        transforms.RandomChoice(
                            [
                                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                            ]
                        )
                    ],
                    p=0.5,
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                ],
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean, std=self.config.std)
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize([512, 512]),
            transforms.Resize([self.config.resolution, self.config.resolution]),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean, std=self.config.std)
        ])

        self.ema_lock = Lock()

    def forward(self, images, inference=False, use_ema=False):
        transform = self.eval_transform if inference else self.train_transform
        processed_images = [transform(img) for img in images]
        batch_tensor = torch.stack(processed_images)
        batch_tensor = batch_tensor.to(self.device, dtype=self.dtype)

        if use_ema:
            with self.ema_lock:
                self.ema_wrapper.copy_ema_to(self.trainable_params, store_temp=True)
                try:
                    output = self.model(batch_tensor)
                finally:
                    self.ema_wrapper.copy_temp_to(self.trainable_params)
        else:
            output = self.model(batch_tensor)
        
        return output

    @torch.no_grad()
    def get_reward(self, images):
        self.model.eval()

        is_video = False
        if isinstance(images, list) and len(images) > 0:
            first_item = images[0]
            if isinstance(first_item, np.ndarray) and first_item.ndim == 4:
                is_video = True

        if is_video:
            batch_pil_images = []
            for video_arr in images:
                F, H, W, C = video_arr.shape
                frame_indices = [0, F // 2, F - 1]
                for idx in frame_indices:
                    frame = video_arr[idx] # (H, W, C)
                    batch_pil_images.append(Image.fromarray(frame))
            
            output = self.forward(batch_pil_images, inference=True, use_ema=True)
            probs_fake = output['prob'].detach() # tensor shape (B*3,)
            rewards_flat = 1.0 - probs_fake
            rewards = rewards_flat.view(-1, 3).mean(dim=1).cpu().tolist()
            
            return rewards

        else:
            output = self.forward(images, inference=True, use_ema=True)
            probs_fake = output['prob'].detach().cpu().tolist()
            rewards = [1.0 - p for p in probs_fake]  # Detection model returns false confidence scores

        return rewards

    def update_ema(self):
        self.ema_wrapper.step(self.trainable_params, self.global_step_d)
        self.global_step_d += 1


def train_discriminator_step(trainable_scorer, current_real_batch, current_fake_batch, historical_batch, optimizer, accelerator):
    trainable_scorer.train()

    if historical_batch:
        hist_real_batch, hist_fake_batch = zip(*historical_batch)
        real_images = list(current_real_batch) + list(hist_real_batch)
        fake_images = list(current_fake_batch) + list(hist_fake_batch)
    else:
        real_images, fake_images = current_real_batch, current_fake_batch

    if isinstance(trainable_scorer, DDP):
        model_instance = trainable_scorer.module.model 
    else:
        model_instance = trainable_scorer.model

    out_real = trainable_scorer(real_images, use_ema=False)
    logits_real = out_real['cls'] if isinstance(out_real, dict) else out_real
    
    out_fake = trainable_scorer(fake_images, use_ema=False)
    logits_fake = out_fake['cls'] if isinstance(out_fake, dict) else out_fake
    
    targets_real = torch.tensor([[0.95, 0.05]] * logits_real.size(0), device=accelerator.device)
    targets_fake = torch.tensor([[0.05, 0.95]] * logits_fake.size(0), device=accelerator.device)
    
    criterion = nn.CrossEntropyLoss()

    if hasattr(model_instance, 'get_losses'):
        loss_dict_real = model_instance.get_losses(
            pred_dict=out_real,
            labels=targets_real, 
            expert_domain_labels=None,
            criterion=criterion
        )
        loss_dict_fake = model_instance.get_losses(
            pred_dict=out_fake, 
            labels=targets_fake, 
            expert_domain_labels=None, 
            criterion=criterion
        )    
        loss = (loss_dict_real['overall_loss'] + loss_dict_fake['overall_loss']) / 2.0
    
    else:
        loss_real = criterion(logits_real, targets_real)
        loss_fake = criterion(logits_fake, targets_fake)
        loss = (loss_real + loss_fake) / 2.0

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if isinstance(trainable_scorer, DDP):
        trainable_scorer.module.update_ema()
    else:
        trainable_scorer.update_ema()
    
    with torch.no_grad():
        acc = ((logits_real.argmax(1) == 0).float().mean() + (logits_fake.argmax(1) == 1).float().mean()) / 2.0
        
    return loss.item(), acc.item()