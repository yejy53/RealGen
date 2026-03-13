from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import json

class PickScoreScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        processor_path = "../RealGen/models/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "../RealGen/models/PickScore_v1"
        self.device = device
        self.dtype = dtype
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(device)
        self.model = self.model.to(dtype=dtype)
        
    @torch.no_grad()
    def __call__(self, images, prompt):
        # Preprocess images
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}
        # Preprocess text
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}
        
        # Get embeddings
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
        
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        
        # Calculate scores
        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs @ image_embs.T)
        scores = scores.diag()
        # norm to 0-1
        scores = scores
        return scores.tolist()


# Usage example
def main():
    scorer = PickScoreScorer(
        device="cuda",
        dtype=torch.float32
    )
    img_list = [
        "xxx",
        "xxx"
    ]
    image_paths = [Image.open(img) for img in img_list]

    prompts = ["a woman", "woman"]

    print(scorer(image_paths, prompts))

if __name__ == "__main__":
    main()