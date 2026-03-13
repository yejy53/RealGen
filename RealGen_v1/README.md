
## 🚀 Quick Started
### 1. Environment Set Up
Diffusion model Training Framework Based on Flow GRPO：Environment Configuration Reference [Flow GRPO](https://github.com/yifan123/flow_grpo)
```bash
cd /RealGen/RealGen_v1/flow_grpo
conda create -n flow_grpo python=3.10.16
pip install -e .
```
### 2. Model Download
Please download the required models in advance.
- T2I Models：
  - FLux: black-forest-labs/FLUX.1-dev
  - SD: stabilityai/stable-diffusion-3.5-large
  - Other diffusion models
- Reward Models：
  - Detection Model: [Forensic-chat and OmniAID](https://huggingface.co/lokiz666/Realgen-detection-models) or other Fake detection models  
  - Alignment Model: Longclip, clip or other alignment models
### 3. Reward Preparation
The steps above strictly cover the installation of the core repository. Given that different reward models often depend on conflicting library versions, merging them into a single Conda environment can lead to compatibility issues. To mitigate this, please create a new Conda virtual environment and install the corresponding dependencies according to the instructions in [Reward Server](https://github.com/yifan123/reward-server)
```bash
cd /RealGen/RealGen_v1/flow_grpo/reward-server
conda create -n reward_server python=3.10.16
conda activate reward_server
pip install -e .
```
We trained task-specific detectors to serve as reward model based on an existing fake detection models. To clarify, we found that reward hacking occurs easily during GRPO training. The existing detection models tends to give high scores to noisy or blurry images. For this reason, we retrained OmniAID to make it suitable for our task:
- **Semantic Detector**: Forensic-Chat, a generalizable and interpretable detector optimized from Qwen2.5-VL-7B. It assesses authenticity by analyzing image content (e.g., smooth greasy skin, artifacts in faces/hands, unnatural background blur). 
- **Feature Detector**: OmniAID achieves stable and accurate detection by being pre-trained on large-scale real and synthetic datasets. Feature-level artifacts are primarily associated with frequency artifacts and abnormal noise patterns. 

An 8-GPU H200 training node was employed for this study, with seven GPUs allocated for the GRPO training process and one GPU reserved for hosting the reward server. Reference code for running the service:
```bash
CUDA_VISIBLE_DEVICES=7 nohup gunicorn --workers 1 --bind 127.0.0.1:18085 "app_forensic_chat:create_app()" > reward_forensic_chat.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup gunicorn --workers 1 --bind 127.0.0.1:18087 "app_omniaid:create_app()" > reward_omniaid.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup gunicorn --workers 1 --bind 127.0.0.1:18089 "app_longclip:create_app()" > reward_longclip.log 2>&1 &
```
### 4. Start Training GRPO
Model parameter settings are located in `/RealGen/RealGen_v1/flow_grpo/config`, while the main files and training settings are in `/RealGen/RealGen_v1/flow_grpo/scripts`. Notably, we have also updated [GRPO-Guard](https://jingw193.github.io/GRPO-Guard/) to improve the capability of generating high-quality images. Below is a reference for running a selected model:
```bash
cd /RealGen/RealGen_v1/flow_grpo
conda activate flow_grpo
bash scripts/single_node/fast_grpo_flux_guard.sh
```
Additionally, if there are no environmental conflicts and GPU memory is sufficient, the reward function does not need to be deployed as a separate service. It can be modified directly in `/RealGen/RealGen_v1/flow_grpo/flow_grpo/rewards.py`. You may also refer to Flow GRPO.

The dataset is located in `/RealGen/RealGen_v1/flow_grpo/dataset/realgen`. The training set contains short prompts and their rewritten long captions covering multiple topics, such as people, animals, and architecture.
