from hpsv3 import HPSv3RewardInferencer
import os
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
    def __call__(self, prompt, images):
        images_base64 = [pil_image_to_base64(image) for image in images]
        rewards = self.inferencer.reward(images_base64, prompt)
        scores = [reward[0].item() for reward in rewards]
        scores = [num / 15 for num in scores]
        return scores
    


def main():
    scorer = HPSv3Scorer(
        device="cuda"
    )
    image_paths = ["xxx", "xxx"]
    prompts = [
        "cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker",
        "cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker"
    ]
    image_paths = [Image.open(img) for img in image_paths]

    data = {
        "images": image_paths,
        "prompts": prompts,
    }

    images = data["images"]
    prompts = data["prompts"]

    print(scorer(prompts, images))

if __name__ == "__main__":
    main()