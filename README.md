# RealGen: Photorealistic Text-to-Image Generation via Detector-Guided Rewards
<p align="center">
  <a href="https://arxiv.org/abs/2512.00473" target="_blank"><img src="https://img.shields.io/badge/arXiv-arXiv-red?style=badge&logo=arXiv" alt="Paper PDF" height="25"></a>
  <a href='https://yejy53.github.io/RealGen/'><img src='https://img.shields.io/badge/Project_Page-RealGen-green' height="25"></a>
  <a href="https://huggingface.co/Yunncheng/RealGen-V2/tree/main" target="_blank">
    <img src="https://img.shields.io/badge/Model_Weights-HuggingFace-orange?style=badge&logo=huggingface&logoColor=white" alt="Model Weights" height="25">
  </a>
</p>

## 📰 News
* **[2026.03.13]**  🔥 We have released **RealGen-V2**, introducing a dynamic GAN-inspired online RL paradigm! Check out the **[** [RealGen-V2 Documentation](RealGen_v2/README.md) **]**
* **[2025.12.02]**  🔥 We have released **RealGen: Photorealistic Text-to-Image Generation via Detector-Guided Rewards**. Check out the **[** [Paper](https://arxiv.org/abs/2512.00473);  **]**. 

![fig1](figures/fig1.png)


## 🏆 Contributions

* ✨ **What did we do?** We propose **RealGen**, a text-to-image generator capable of producing highly convincing photorealistic images. It leverages a Detector Reward-guided GRPO post-training to escape detector identification, thereby reducing artifacts and enhancing image realism and detail.
* 📐 **How to evaluate performance?** We introduce **RealBench**, a new benchmark for evaluating photorealism that achieves human-free automated scoring through Detector-Scoring and Arena-Scoring.
* 🔧 **How effective was it?** RealGen significantly outperforms both general image models (like GPT-Image-1, Qwen-Image) and specialized realistic models (like FLUX-Krea) in realism, details, and aesthetics on the T2I task.

<img src="figures/RealGEN-Comparison.jpg" width="600" alt="fig2">

## 🤝 Concurrent Work

We are pleased to find that the strategy of utilizing **AIGC detectors as reward signals** has been independently explored by other excellent concurrent works. We acknowledge and recommend checking out:

* **[LongCat-Image](https://github.com/meituan-longcat/LongCat-Image/)**: They innovatively incorporate an AIGC detection model as a reward during the RL phase, utilizing adversarial signals to guide the model toward generating images with the **texture and fidelity of the real physical world**.
* **[Z-Image](https://github.com/Tongyi-MAI/Z-Image)**: In their RLHF pipeline, they design a comprehensive reward model where **AI-Content Detection perception** serves as a critical dimension, alongside instruction-following capability and aesthetic quality.

It is exciting to see the community converging on this effective paradigm to bridge the gap between generated and real distributions.


## 🖼️ Visual Comparison

### ✨ RealGen-V2

<table style="text-align: center; width: 100%;">
  <tr>
    <th colspan="4" style="font-size: 1.1em; padding-top: 10px;">Z-Image (Baseline)</th>
  </tr>
  <tr>
    <td width="25%"><img src="figures/realgenv2_comparsion/z-image-0.jpg" alt="Baseline 0"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/z-image-1.jpg" alt="Baseline 1"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/z-image-2.jpg" alt="Baseline 2"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/z-image-3.jpg" alt="Baseline 3"></td>
  </tr>
  <tr>
    <td width="25%"><img src="figures/realgenv2_comparsion/z-image-4.jpg" alt="Baseline 4"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/z-image-5.jpg" alt="Baseline 5"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/z-image-6.jpg" alt="Baseline 6"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/z-image-7.jpg" alt="Baseline 7"></td>
  </tr>
  <tr>
    <th colspan="4" style="font-size: 1.1em; padding-top: 15px;">After RealGen-V2 (Ours)</th>
  </tr>
  <tr>
    <td width="25%"><img src="figures/realgenv2_comparsion/realgen-0.jpg" alt="Ours 0"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/realgen-1.jpg" alt="Ours 1"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/realgen-2.jpg" alt="Ours 2"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/realgen-3.jpg" alt="Ours 3"></td>
  </tr>
  <tr>
    <td width="25%"><img src="figures/realgenv2_comparsion/realgen-4.jpg" alt="Ours 4"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/realgen-5.jpg" alt="Ours 5"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/realgen-6.jpg" alt="Ours 6"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/realgen-7.jpg" alt="Ours 7"></td>
  </tr>
</table>

<br>

<table style="text-align: center; width: 100%;">
  <tr>
    <th colspan="4" style="font-size: 1.1em; padding-top: 10px;">Z-Image (Baseline)</th>
  </tr>
  <tr>
    <td width="25%"><img src="figures/realgenv2_comparsion/z-image-8.jpg" alt="Baseline 8"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/z-image-9.jpg" alt="Baseline 9"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/z-image-10.jpg" alt="Baseline 10"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/z-image-11.jpg" alt="Baseline 11"></td>
  </tr>
  <tr>
    <td width="25%"><img src="figures/realgenv2_comparsion/z-image-12.jpg" alt="Baseline 12"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/z-image-13.jpg" alt="Baseline 13"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/z-image-14.jpg" alt="Baseline 14"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/z-image-15.jpg" alt="Baseline 15"></td>
  </tr>
  <tr>
    <th colspan="4" style="font-size: 1.1em; padding-top: 15px;">After RealGen-V2 (Ours)</th>
  </tr>
  <tr>
    <td width="25%"><img src="figures/realgenv2_comparsion/realgen-8.jpg" alt="Ours 8"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/realgen-9.jpg" alt="Ours 9"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/realgen-10.jpg" alt="Ours 10"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/realgen-11.jpg" alt="Ours 11"></td>
  </tr>
  <tr>
    <td width="25%"><img src="figures/realgenv2_comparsion/realgen-12.jpg" alt="Ours 12"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/realgen-13.jpg" alt="Ours 13"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/realgen-14.jpg" alt="Ours 14"></td>
    <td width="25%"><img src="figures/realgenv2_comparsion/realgen-15.jpg" alt="Ours 15"></td>
  </tr>
</table>


### ✨ RealGen-V1

![fig1](figures/fig7.png)




## 🚀 Quick Started
- It should be stated that our proposed detection-for-generation framework is compatible with all diffusion-model-based GRPO paradigms, such Dance GRPO and Flow GRPO. 

### RealGen-V1
For the original V1 implementation, please refer to:  
👉 **[RealGen-V1 Documentation](RealGen_v1/README.md)**

### RealGen-V2
For the latest V2 implementation, please refer to:  
👉 **[RealGen-V2 Documentation](RealGen_v2/README.md)**

## 📊 Evaluation
The inference and evaluation processes are realized according to the code in `/RealGen/eval`.

## 🤗 Acknowledgement
This repo is based on [Flow GRPO](https://github.com/yifan123/flow_grpo). We thank the authors for their valuable contributions to the AlGC community.

## 📕 BibTeX 

```bib
@article{ye2025realgen,
  title={RealGen: Photorealistic Text-to-Image Generation via Detector-Guided Rewards},
  author={Ye, Junyan and Zhu, Leqi and Guo, Yuncheng and Jiang, Dongzhi and Huang, Zilong and Zhang, Yifan and Yan, Zhiyuan and Fu, Haohuan and He, Conghui and Li, Weijia},
  journal={arXiv preprint arXiv:2512.00473},
  year={2025}
}
```
