from setuptools import setup, find_packages

setup(
    name="flow-grpo",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        "transformers==4.57.0",
        "accelerate==1.4.0",
        "diffusers==0.36.0", 
        
        "ftfy==6.3.1"
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scipy==1.15.2",
        "scikit-learn==1.6.1",
        "scikit-image==0.25.2",
        "paddlepaddle-gpu",
        "paddleocr==2.7.3",
        "Levenshtein",
        "imageio-ffmpeg==0.6.0",
        
        "albumentations==1.4.10",  
        "opencv-python==4.6.0.66",
        "pillow==10.4.0",
        
        "tqdm==4.67.1",
        "wandb==0.18.7",
        "pydantic==2.10.6",  
        "requests==2.32.3",
        "matplotlib==3.10.0",
        
        # "flash-attn==2.7.4.post1",
        "deepspeed==0.15.4",  
        "peft==0.17.0",       
        "bitsandbytes==0.47.0",
        
        "aiohttp==3.11.13",
        "fastapi==0.115.11", 
        "uvicorn==0.34.0",
        
        "huggingface-hub==0.36.0",  
        "datasets==3.3.2",
        "tokenizers==0.22.1",
        
        "einops==0.8.1",
        "nvidia-ml-py==12.570.86",
        "xformers",
        "absl-py",
        "ml_collections",
        "sentencepiece",
        "openai",
    ],
    extras_require={
        "dev": [
            "ipython==8.34.0",
            "black==24.2.0",
            "pytest==8.2.0"
        ]
    }
)
