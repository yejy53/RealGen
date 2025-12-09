import os
import re
import torch
import torch.distributed as dist
from pathlib import Path
from torch.utils.data import Dataset, DistributedSampler
from diffusers import StableDiffusion3Pipeline, StableDiffusionPipeline, EulerDiscreteScheduler, DiffusionPipeline
from safetensors.torch import load_file
import json
from PIL import Image
import torchvision.transforms as T
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from peft import PeftModel 

class PromptDataset(Dataset):
    def __init__(self, file_path):
        if not file_path is None:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [line.strip() for line in f]
            self.prompts = data
            
        else:
            self.prompts = [
                'portrait of a woman',
                'A realistic portrait of a young Chinese woman, clear facial features, smooth black hair, natural makeup, wearing casual clothes, photographed in soft natural light.',
                'A candid portrait of a young Chinese woman, slightly uneven skin with visible pores and small blemishes, faint dark circles under the eyes, smudged or slightly imperfect makeup, loose strands of black hair falling naturally, photographed with clear focus from eye-level camera angle, capturing raw emotion and a natural, everyday expression in soft daylight.',
                'A realistic portrait of a young woman, clear facial features, smooth black hair, natural makeup, wearing casual clothes, photographed in soft natural light.',
                'A candid portrait of a young woman, slightly uneven skin with visible pores and small blemishes, faint dark circles under the eyes, smudged or slightly imperfect makeup, loose strands of black hair falling naturally, photographed with clear focus from eye-level camera angle, capturing raw emotion and a natural, everyday expression in soft daylight.'
            ]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

def sanitize_filename(text, max_length=200):
    sanitized = re.sub(r'[\\/:*?"<>|]', '_', text)
    return sanitized[:max_length].rstrip() or "untitled"
    # --node_rank $NODE_RANK \

def distributed_setup():
    try:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    except Exception as e:
        rank =0
        world_size=8
    local_rank = int(os.environ['LOCAL_RANK'])

    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def main():
    rank, local_rank, world_size = distributed_setup()

    model_path = ""
    lora_path = ""
    dataset_path = "/short-img/banchmark/short_prompts.txt"
    output_path = "/short-img/xxx/photo"
    
    pipe = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path)
    pipe.transformer.set_adapter("default")

    pipe = pipe.to("cuda")
    dataset = PromptDataset(dataset_path)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx in sampler:
        prompt = dataset[idx]
        try:
            generator = torch.Generator(device=f"cuda:{local_rank}")
            generator.manual_seed(42 + idx*1000)
            image = pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                max_sequence_length=512,
                generator=generator
            ).images[0]

            filename = sanitize_filename(prompt)
            save_path = output_dir / f"{idx:05}.jpg"
            image.save(save_path)
            print(f"[Rank {rank}] Generated: {save_path.name}")

        except Exception as e:
            print(f"[Rank {rank}] Error processing '{prompt[:20]}...': {str(e)}")

if __name__ == "__main__":
    main()
