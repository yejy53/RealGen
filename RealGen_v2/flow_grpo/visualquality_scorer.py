from PIL import Image
import torch
import re
import base64
from io import BytesIO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    return base64_qwen

def extract_scores(output_texts):
    scores = []
    for model_output in output_texts:
        try:
            model_output_matches = re.findall(r'<answer>(.*?)</answer>', model_output, re.DOTALL)
            model_answer = model_output_matches[-1].strip() if model_output_matches else model_output.strip()
            
            score_match = re.search(r'\d+(\.\d+)?', model_answer)
            if score_match:
                score = float(score_match.group())
            else:
                raise ValueError("No number found in answer")
            score = max(1.0, min(5.0, score))
            
        except Exception as e:
            print(f"Warning: Failed to parse score from VisualQuality-R1. Defaulting to 1.0.")
            score = 1.0 
        
        scores.append(score)
    return scores

class VisualQualityScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.bfloat16, max_batch_size = 4):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        
        model_path = "VisualQuality-R1-7B"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            attn_implementation="flash_attention_2",
            device_map=self.device,
        )
        self.model.requires_grad_(False)
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "left"

        self.prompt_text = (
            "You are doing the image quality assessment task. Here is the question: "
            "What is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, "
            "rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality."
        )

        self.question_template = "{Question} Please only output the final answer with only one score in <answer> </answer> tags."

    @torch.no_grad()
    def __call__(self, images):
        """
        Args:
            images: List[PIL.Image]
        Returns:
            rewards: List[float]
        """
        if not isinstance(images, list):
            images = [images]
        
        all_rewards = []

        for i in range(0, len(images), self.max_batch_size):
            batch_images = images[i : i + self.max_batch_size]
            
            # Construct Batch Messages
            messages = []
            for image in batch_images:
                base64_qwen = pil_image_to_base64(image)
                message = [
                    {
                        "role": "user",
                        "content": [
                            {'type': 'image', 'image': base64_qwen},
                            {"type": "text", "text": self.question_template.format(Question=self.prompt_text)},
                        ],
                    }
                ]
                messages.append(message)

            # Preprocess
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, add_vision_id=True)
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

            # Generation
            generated_ids = self.model.generate(
                **inputs, 
                use_cache=True, 
                max_new_tokens=512,
                do_sample=True,
                top_k=50,
                top_p=1
            )

            # Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # Extract scores
            batch_rewards = extract_scores(batch_output_text)
            all_rewards.extend(batch_rewards)

        return all_rewards


# Usage example
def main():

    scorer = VisualQualityScorer(
        device="cuda",
        dtype=torch.bfloat16
    )
    
    images=['',
        ''
    ]

    pil_images = [Image.open(img) for img in images]

    print("Inference starting...")
    scores = scorer(pil_images)
    print(f"Scores: {scores}")
        

if __name__ == "__main__":
    main()