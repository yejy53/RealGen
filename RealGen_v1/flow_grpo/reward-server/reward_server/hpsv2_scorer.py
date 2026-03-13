# Based on https://github.com/RE-N-Y/imscore/blob/main/src/imscore/preference/model.py

from importlib import resources
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from transformers import AutoImageProcessor,CLIPProcessor, CLIPModel
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import numpy as np
import random
from PIL import Image

# Some AI-Look magic word 
# `Painting' is most powerful word to elimate the oily texture and drop style ability. We combine it with other word to get more stable result.
def get_random_cg_oily_adjective(index = 0):
    cg_oily_adjectives = [
        "Concept art",
        "Painting",
        "Anime",
        "Flat",
        "Oil"
        # "3D",
        # "Photo",
        # "Dark",
        # "Blue",
        # "Concept art"
        # "3D"
        # "Flat-lighting"
        # "Minimalistic",
        # "Cartoonish"
    ]
    return cg_oily_adjectives[index%len(cg_oily_adjectives)]

# Some texture word
def get_random_realism_adjective(index = 0):
    realism_adjectives = [
      "Natural-lighting","Detail","Detailed","Real"
    ]
    return   realism_adjectives[index%len(realism_adjectives)]


class HPS(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        hpsv2_model,hpsv2_token,hpsv2_pre=self.build_reward_model()
        self.model = hpsv2_model.to(dtype=dtype)
        self.token = hpsv2_token 

        #### differentiable preprocessor
        image_mean = (0.48145466, 0.4578275, 0.40821073)
        image_std = (0.26862954, 0.26130258, 0.27577711)
        crop_size = 224
        resize_size = 224 
        def _transform():
            transform = Compose([
                Resize(resize_size, interpolation=BICUBIC),
                CenterCrop(crop_size),

                Normalize(std=image_std,mean=image_mean),
            ])
            return transform
        self.vis_pre = _transform()
        self.device =device

    def build_reward_model(self):
        model, preprocess_train, reprocess_val = create_model_and_transforms(
            'ViT-H-14',
            '../RealGen/models/open_clip_pytorch_model.bin',
            precision='amp',
            device=self.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )

        # Convert device name to proper format
        if isinstance(self.device, int):
            ml_device = str(self.device)
        else:
            ml_device = self.device

        # if ml_device.type != 'cuda':
        #     ml_device = f'cuda:{ml_device}' if ml_device.isdigit() else ml_device

        checkpoint = torch.load('../RealGen/models/HPS_v2.1_compressed.pt', map_location=ml_device)
        model.load_state_dict(checkpoint['state_dict'])
        text_processor = get_tokenizer('ViT-H-14')
        reward_model = model.to(self.device)
        reward_model.eval()

        return reward_model, text_processor, preprocess_train

    #### implment the SRP in CFG-like function；we find the （1-k)*neg + k *pos is less stable. therefore, we change it to (1+k)*pos-neg
    def SRP_cfg(self, prompt,neg_prompt,images):
        image = self.vis_pre(images.squeeze(0)).unsqueeze(0).to(device=self.device, non_blocking=True)
        text = self.token(prompt).to(device=self.device, non_blocking=True)
        neg_text = self.token(neg_prompt).to(device=self.device, non_blocking=True)
        with torch.cuda.amp.autocast():
            # Extract image features and text features for positive and negative prompts
            image_features = self.model.encode_image(image, normalize=True)
            text_features = self.model.encode_text(text, normalize=True)
            text_features_neg = self.model.encode_text(neg_text, normalize=True)
            
            # Compute the reward based on the similarity
            logits_per_image = image_features @ (text_features.T-text_features_neg.T)
            hps_score = torch.diagonal(logits_per_image)
        return hps_score

    def SRP(self, prompt,images):
        image = self.vis_pre(images.squeeze(0)).unsqueeze(0).to(device=self.device, non_blocking=True)
        text = self.token(prompt).to(device=self.device, non_blocking=True)
        with torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image, normalize=True)
            text_features = self.model.encode_text(text, normalize=True)
            logits_per_image = image_features @ (text_features.T)
            hps_score = torch.diagonal(logits_per_image)
        return hps_score    

class Hpsv2Scorer(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model = HPS(self.device).to(self.device)
        self.model.eval()
        self.preprocess = T.Compose([
            T.ToTensor(),
        ])

    @torch.no_grad()
    def __call__(self, images, prompts, step=0): #
        
        rewards = []
        for image, base_prompt in zip(images, prompts):
            step = random.randint(0, 1000)
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            pos_control = get_random_realism_adjective(step)
            # neg_control = get_random_cg_oily_adjective(step)
            # final_positive_prompt = f"{pos_control} and not {neg_control}. {base_prompt}"
            # final_negative_prompt = [f"{neg_control}. {base_prompt}"]
            reward = self.model.SRP(
                prompt=[base_prompt],
                # prompt=[final_positive_prompt],
                images=image_tensor
            ).detach().cpu().tolist()
            rewards.append(reward[0])
        return rewards


def main():
    scorer = Hpsv2Scorer(
        device='cuda'
    )

    images=[
        "xxx",
        "xxx"
    ]
    images = [Image.open(img) for img in images]
    prompts=[
        "portrait of a man",
        "A realistic portrait of a young woman, clear facial features, smooth black hair, natural makeup, wearing casual clothes, photographed in soft natural light."
    ]

    print(scorer(images, prompts))

if __name__ == "__main__":
    main()