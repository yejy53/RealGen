from importlib import resources
import torch
import pickle
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from transformers import AutoImageProcessor,CLIPProcessor, CLIPModel
import numpy as np
from PIL import Image
from Long_CLIP.model import longclip
import os
from tqdm import tqdm

class ClipScorer(torch.nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model, self.preprocess = longclip.load("../RealGen/models/LongCLIP-L/longclip-L.pt", device=self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, images, prompts):
        results = []
        for prompt, image in tqdm(zip(prompts, images)):
            text = longclip.tokenize(prompt).to(self.device)
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(text)
                
                logits_per_image = image_features @ text_features.T
                probs = logits_per_image.diagonal()
            results.append(probs.item())
        return results



def main():
    scorer = ClipScorer(
        device='cuda',
        dtype=torch.bfloat16
    )

    is_rewrite ="short"
    img_list = ['xxx']
    for img_path in img_list:
        with open(f'/model-eval/{is_rewrite}-img/banchmark/{is_rewrite}_prompts.txt', 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f]

        image_paths=[]
        count = 0
        for i in range(1041):
            
            path = f"/model-eval/{is_rewrite}-img/{img_path}/photo/{i:05}.jpg"
            if os.path.exists(path):
                image_paths.append(path)
            else:
                prompts.pop(i-count)
                count+=1
        image_paths = [Image.open(img) for img in image_paths]

        print(f"=========={img_path}==========")
        result = scorer(image_paths, prompts)
        print(sum(result)/len(result))

if __name__ == "__main__":
    main()
