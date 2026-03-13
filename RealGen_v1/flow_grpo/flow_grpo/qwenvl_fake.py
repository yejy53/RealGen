from PIL import Image
import torch
import re
import base64
from io import BytesIO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    return base64_qwen

def extract_scores(output_logits, processor):
    vocab = processor.tokenizer.get_vocab()
    probabilities = output_logits[0, -1, :]
    # probabilities = F.softmax(last_token_logits, dim=-1)
    probabilities_ = probabilities.float().cpu().numpy()
    # fake_score = max(probabilities_[vocab['fake']], probabilities_[vocab['Fake']])
    # real_score = max(probabilities_[vocab['Real']], probabilities_[vocab['real']])
    fake_score = (probabilities_[vocab['fake']] + probabilities_[vocab['Fake']])/2
    real_score = (probabilities_[vocab['Real']] + probabilities_[vocab['real']])/2
    compare_score = np.array([fake_score, real_score])
    e_x = np.exp(compare_score - np.max(compare_score))
    score = e_x / e_x.sum()
    return score[1]

class QwenVLScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "../RealGen/models/Forensic-Chat",
            torch_dtype=self.dtype,
            attn_implementation="flash_attention_2",
            device_map=None,
        ).to(self.device)
        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained("../RealGen/models/Forensic-Chat", use_fast=True)
        
    @torch.no_grad()
    def __call__(self, images):
        images_base64 = [pil_image_to_base64(image) for image in images]
        rewards = []
        for base64_qwen in images_base64:
            messages=[]
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": base64_qwen},
                        {"type": "text", "text": (
                            "Analyze the provided image. "
                            "Decide whether it is a real photograph or AI-generated. "
                            "The first word must be either 'real' or 'fake'."
                        )},
                    ],
                },
            ])

            # Preparation for batch inference
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits

            reward = extract_scores(logits, self.processor)
            rewards.append(reward)
        return rewards
    

# Usage example
def main():
    scorer = QwenVLScorer(
        device="cuda",
        dtype=torch.bfloat16
    )
    images=['xxx']
    pil_images = [Image.open(img) for img in images]

    print(scorer(pil_images))

if __name__ == "__main__":
    main()