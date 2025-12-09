from hpsv3 import HPSv3RewardInferencer
import os
import json
from tqdm import tqdm
import torch
from io import BytesIO
import base64
from PIL import Image

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    return base64_qwen

class HPSv3Scorer(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.inferencer = HPSv3RewardInferencer(checkpoint_path="../RealGen/models/HPSv3.safetensors", device=self.device)
        
    @torch.no_grad()
    def __call__(self, prompts, images):
        result = []
        for prompt, image in tqdm(zip(prompts, images)):
            images_base64 = pil_image_to_base64(image)
            reward = self.inferencer.reward([images_base64], [prompt])
            score = reward[0][0].item()
            result.append(score)
        return result
    

def main():
    scorer = HPSv3Scorer(
        device="cuda"
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
        result = scorer(prompts, image_paths)
        print(sum(result)/len(result))

if __name__ == "__main__":
    main()
