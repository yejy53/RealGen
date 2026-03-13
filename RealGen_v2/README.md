# RealGen-V2

## 📖 Overview
Welcome to the official repository for **RealGen-V2**, a major architectural upgrade to our RealGen framework.

While **RealGen-V1** successfully enhanced generation realism using AI-generated image detector, it relied static detector reward models. This V1 paradigm ultimately faced inherent bottlenecks, such as reward hacking or eventual mode collapse, because the generator would learn to exploit the frozen evaluation metrics.

**RealGen-V2** addresses these limitations by shifting from a static environment to a dynamic, adversarial loop. We introduce a **GAN-inspired RL** paradigm that replaces text-only prompts with **Paired Data** (ground-truth real images + captions). Coupled with an upgraded **Hybrid Reward System** featuring adversarially updatable discriminators, V2 continuously challenges the generator. This evolution from V1 ensures the generator cannot over-optimize a fixed target, resulting in significantly more photorealistic and higher-quality image synthesis.


## 🖼️ Visual Comparison

<table style="text-align: center; width: 100%;">
  <tr>
    <th colspan="4" style="font-size: 1.1em; padding-top: 10px;">Z-Image (Baseline)</th>
  </tr>
  <tr>
    <td width="25%"><img src="../figures/realgenv2_comparsion/z-image-0.jpg" alt="Baseline 0"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/z-image-1.jpg" alt="Baseline 1"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/z-image-2.jpg" alt="Baseline 2"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/z-image-3.jpg" alt="Baseline 3"></td>
  </tr>
  <tr>
    <td width="25%"><img src="../figures/realgenv2_comparsion/z-image-4.jpg" alt="Baseline 4"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/z-image-5.jpg" alt="Baseline 5"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/z-image-6.jpg" alt="Baseline 6"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/z-image-7.jpg" alt="Baseline 7"></td>
  </tr>
  <tr>
    <th colspan="4" style="font-size: 1.1em; padding-top: 15px;">After RealGen-V2 (Ours)</th>
  </tr>
  <tr>
    <td width="25%"><img src="../figures/realgenv2_comparsion/realgen-0.jpg" alt="Ours 0"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/realgen-1.jpg" alt="Ours 1"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/realgen-2.jpg" alt="Ours 2"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/realgen-3.jpg" alt="Ours 3"></td>
  </tr>
  <tr>
    <td width="25%"><img src="../figures/realgenv2_comparsion/realgen-4.jpg" alt="Ours 4"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/realgen-5.jpg" alt="Ours 5"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/realgen-6.jpg" alt="Ours 6"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/realgen-7.jpg" alt="Ours 7"></td>
  </tr>
</table>

<br>

<table style="text-align: center; width: 100%;">
  <tr>
    <th colspan="4" style="font-size: 1.1em; padding-top: 10px;">Z-Image (Baseline)</th>
  </tr>
  <tr>
    <td width="25%"><img src="../figures/realgenv2_comparsion/z-image-8.jpg" alt="Baseline 8"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/z-image-9.jpg" alt="Baseline 9"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/z-image-10.jpg" alt="Baseline 10"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/z-image-11.jpg" alt="Baseline 11"></td>
  </tr>
  <tr>
    <td width="25%"><img src="../figures/realgenv2_comparsion/z-image-12.jpg" alt="Baseline 12"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/z-image-13.jpg" alt="Baseline 13"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/z-image-14.jpg" alt="Baseline 14"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/z-image-15.jpg" alt="Baseline 15"></td>
  </tr>
  <tr>
    <th colspan="4" style="font-size: 1.1em; padding-top: 15px;">After RealGen-V2 (Ours)</th>
  </tr>
  <tr>
    <td width="25%"><img src="../figures/realgenv2_comparsion/realgen-8.jpg" alt="Ours 8"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/realgen-9.jpg" alt="Ours 9"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/realgen-10.jpg" alt="Ours 10"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/realgen-11.jpg" alt="Ours 11"></td>
  </tr>
  <tr>
    <td width="25%"><img src="../figures/realgenv2_comparsion/realgen-12.jpg" alt="Ours 12"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/realgen-13.jpg" alt="Ours 13"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/realgen-14.jpg" alt="Ours 14"></td>
    <td width="25%"><img src="../figures/realgenv2_comparsion/realgen-15.jpg" alt="Ours 15"></td>
  </tr>
</table>


## 📦 Model Zoo
The LoRA weights for the generative Model can be downloaded from the table below. 

> **Note:** We are continuously optimizing the training dynamics and evaluating the performance of RealGen-V2.

| LoRA Weight | Generative Model | Folder Name | Download Link |
| :--- | :--- | :--- | :--- |
| **RealGen-V2 LoRA** | Z-Image | Z-Image-LoRA | [Link](https://huggingface.co/Yunncheng/RealGen-V2/tree/main/Z-Image-LoRA) |


## 📊 Key Differences: V1 vs. V2

| Feature | RealGen-V1 | RealGen-V2 |
| :--- | :--- | :--- |
| **RL Paradigm** | Traditional RL | GAN-inspired Online RL |
| **Data Input** | Text-only Prompts | Paired Data (Caption + Real Image) |
| **Reward System** | Static Rewards | Hybrid (Fixed Rewards + Updatable Detectors) |
| **Detectors** | Frozen Detector | Multiple Updatable Detectors |

## 🏗 Architecture Deep Dive

The V2 architecture introduces a robust adversarial optimization loop between the Image Generator and a newly designed Hybrid Reward System, effectively mitigating the reward hacking commonly observed in static RL setups and continuously promoting generator improvement.

### 1. Paired Data Generation Pipeline
Unlike V1, which relies solely on ungrounded text prompts, V2 utilizes a **Paired Dataset**. The pipeline feeds a caption from `Real Image` into the Image Generator to synthesize a `Fake Image`. This generated output is then explicitly paired with its corresponding ground-truth `Real Image` from the dataset, establishing a rigorous visual reference for subsequent dynamic detector updates.

### 2. Hybrid Reward System
To prevent the generator from over-optimizing against fixed targets, we bifurcate the reward mechanism into two parallel branches:
* **Fixed Reward Models (Static RMs):** A suite of frozen models providing stable optimization signals. These typically include an Aesthetic Scorer for visual fidelity and a Vision-Language Critic for text-image alignment.
* **Multiple Updatable Detectors:** A group of trainable discriminators. Rather than relying on a single frozen detector like in V1, this group of multiple detectors continuously evaluates the evasion probability (escape rate) of the `Fake Images` produced by the current generator. Once the generator's reward signal plateaus or reaches a predefined threshold, the optimization loop pauses the generator and adversarially updates the weights of all detectors in this group using the paired data.

### 3. GRPO Optimization Loop
The aggregated reward signals from both the fixed reward models and the multiple updatable detectors are unified and fed into the **Flow-GRPO** algorithm. This dictates the policy update for the Image Generator, creating a continuous, mutually reinforcing cycle between generation and discrimination.


## 🚀 Quick Started
> **Important Note:** RealGen-V2 depends on a slightly different environment than V1, so you may need to update or reinstall the environment.

### 1. Environment Set Up
Diffusion model Training Framework Based on Flow GRPO：
```bash
cd /RealGen/RealGen_v2/flow_grpo
conda create -n flow_grpo python=3.10.16
pip install -e .
```
### 2. Model Download
Please download the required models in advance.
- T2I Models：
  - FLux: black-forest-labs/FLUX.1-dev
  - Z-Imgae: Tongyi-MAI/Z-Image
  - Other diffusion models
- Reward Models：
  - Detection Model: [OmniAID and OmniAID-DINO](https://github.com/yunncheng/OmniAID) or other Fake detection models  
  - Visual Quality Model: [VisualQuality-R1](https://huggingface.co/TianheWu/VisualQuality-R1-7B) or other aesthetic models
  - Alignment Model: Longclip, clip or other alignment models
### 3. Dataset Download
The prompt dataset is located in `/RealGen/RealGen_v2/flow_grpo/dataset/realgen`. The metadata file `train.jsonl` contains rewritten long captions and relative paths to their corresponding real images, covering multiple topics, such as people, animals, and architecture. Please download the real image from [here](https://huggingface.co/datasets/Yunncheng/RealGen-V2-real-images/tree/main) and then place the `images` directory directly under `/RealGen/RealGen_v2/flow_grpo/dataset/realgen`. 
### 4. Start Training GRPO
Model parameter settings are located in `/RealGen/RealGen_v2/config`, while the main files and training settings are in `/RealGen/RealGen_v2/scripts`. And not that you need to specify the path to the pre-trained weights in the corresponding reward model file (such as `/RealGen/RealGen_v2/flow_grpo/trainable_scorers.py` or `/RealGen/RealGen_v2/flow_grpo/visualquality_scorer.py`).Below is a reference for running a selected model:
```bash
cd /RealGen/RealGen_v2
conda activate flow_grpo
bash scripts/single_node/grpo_zimage_base_realgen.sh
```

