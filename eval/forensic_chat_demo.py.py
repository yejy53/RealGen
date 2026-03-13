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
from peft import PeftModel

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
        results = []
        for base64_qwen in tqdm(images):
            messages=[]
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": base64_qwen},
                        {"type": "text", "text": (
                            "You are a professional Forensic Image Analyst. Inspect this image and determine if it is Fake (AI-generated) or Real (camera-captured). "
                            "Your response must strictly begin with the exact sentence: 'This is a [real/fake] image.' "
                            "And then provide your detailed analysis supporting your conclusion."
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
    
            # chat
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            results.append(output_text[0])

        return results
    

# Usage example
def main():
    scorer = QwenVLScorer(
        device="cuda",
        dtype=torch.bfloat16
    )

    image_paths = [
        "your/image/path"
    ]
    image_paths = [Image.open(img) for img in image_paths]

    result = scorer(image_paths)
    print(result)

if __name__ == "__main__":
    main()
