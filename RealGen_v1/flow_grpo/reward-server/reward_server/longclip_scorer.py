from importlib import resources
import torch
import pickle
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from transformers import AutoImageProcessor,CLIPProcessor, CLIPModel
import numpy as np
from PIL import Image
from reward_server.Long_CLIP.model import longclip
    

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
        for prompt, image in zip(prompts, images):
            text = longclip.tokenize(prompt).to(self.device)
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(text)
                
                logits_per_image = image_features @ text_features.T
                probs = logits_per_image.diagonal()/100
            results.append(probs.item())
        return results

def main():
    scorer = ClipScorer(
        device='cuda',
        dtype=torch.bfloat16
    )

    prompts = [
        "A young Asian woman with pigtails and fuzzy hair ties sits by a pool in a blue one-piece swimsuit, hugging her knees, at a garden pool surrounded by tall palm trees and stone steps. She wears a light blue one-piece swimsuit with a small chest logo, a slim bracelet, and a tiny nose ring, and a delicate script tattoo runs along her upper arm; the fuzzy hair ties loop around her wrists as she leans forward, her lips relaxed and her eyes soft. The scene shows her bare legs tucked close, her cheeks gently flushed, with fine pores and faint under-eye shadows visible in the close light, and a few flyaway strands escaping her tidy pigtails. Behind her, the water ripples with pale highlights and soft reflections, a pale building facade and a low arched bridge sit out of focus, and the path recedes into a cool, shaded background that deepens to a teal-blue. The mood feels calm and private, framed from a slightly low angle with shallow depth of field and bright daylight diffused by a thin overcast, carrying a quiet, candid snapshot quality with slight noise; captured on a Fujifilm camera, RAF file.",
        "An Asian woman in a sailor uniform cosplay with cat ears and pigtails sits in the shallow water of a beach at golden hour, her knees drawn up while she looks toward the camera with a calm, attentive expression. Soft, low sun skims across the rippled water and paints a warm band along the shore, while the background falls into a creamy blur of pale sand, scattered pebbles, and distant greenery. Her hair is pulled into a high ponytail with a few stray strands escaping near the ear, and her skin shows faint pores with slight dryness around the cheeks, lending a natural, unretouched realism. She wears a white sailor uniform with a black tie and navy placket, a small chest pocket, and a black belt with a white buckle; a thin chain necklace and a delicate bracelet add subtle highlights, and the translucent straps of her patterned bikini top peek through at the shoulders. Her left hand rests on the sand with a slim red nail, the toes of her white flip-flops peeking out, and the water beads lightly on her calves as it laps against the shore. The scene reads as a candid beach portrait with shallow depth of field and soft, pastel-muted tones, a touch of grain and gentle motion blur in the rippling surf, taken on a Fujifilm camera, saved as DSCF2045.RAF."
    ]
    image_paths = [
        "xxx",
        "xxx"
    ]
    images = [Image.open(img) for img in image_paths]
    print(scorer(images, prompts))

if __name__ == "__main__":
    main()