# From patchSAE to Neroun Amplication and TTA

> Original Authors of the project is  https://github.com/dynamical-inference/patchsae

### *Additional works were done at section TTA 

<div align="center">
    <img width="800" alt="PatchSAE visualization" src="./assets/sae_arch.gif">
</div>

## 🚀 Quick Navigation

- [Getting Started](#-getting-started)
- [Training & Analysis](#-patchsae-training-and-analysis)
- [Test-Time-Adaptation(Parkprogrammer addtion)](#test-time-adaptation-with-neuron-amplification)
- [License & Credits](#-license--credits)

## 🛠 Getting Started

Set up your environment with these simple steps:

```bash
# Create and activate environment
conda create --name patchsae python=3.12
conda activate patchsae

# Install dependencies
pip install -r requirements.txt

# Always set PYTHONPATH before running any scripts
cd patchsae
PYTHONPATH=./ python src/demo/app.py
```


#### 1. Setup Local Demo

First, download the necessary files:

- [out.zip](https://drive.google.com/file/d/1NJzF8PriKz_mopBY4l8_44R0FVi2uw2g/edit)
- [data.zip](https://drive.google.com/file/d/1reuDjXsiMkntf1JJPLC5a3CcWuJ6Ji3Z/edit)

You can download the files using `gdown` as follows:

```bash
# Activate environment first (see Getting Started)

# Download necessary files (35MB + 513MB)
gdown --id 1NJzF8PriKz_mopBY4l8_44R0FVi2uw2g  # out.zip
gdown --id 1reuDjXsiMkntf1JJPLC5a3CcWuJ6Ji3Z  # data.zip

# Extract files
unzip data.zip
unzip out.zip
```

> 💡 Need `gdown`? Install it with: `conda install conda-forge::gdown`

Your folder structure should look like:

```
patchsae/
├── configs/
├── data/      # From data.zip
├── out/       # From out.zip
├── src/
│   └── demo/
│       └── app.py
├── tasks/
├── requirements.txt
└── ... (other files)
```

⚠️ **Note**:
- First run will download datasets from HuggingFace automatically (About 30GB in total)
- Demo runs on CPU by default
- Access the interface at http://127.0.0.1:7860 (or the URL shown in terminal)

## 📊 PatchSAE Training and Analysis

- **Training Instructions**: See [tasks/README.md](./tasks/README.md)
- **Analysis Notebooks**:
  - [demo.ipynb](./demo.ipynb)
  - [analysis.ipynb](./analysis/analysis.ipynb)


## 😊 Test-Time-Adaptation with Neuron Amplification
- run `run_tta.py` for evaluation of Neuron Amplication
- Implementation Wrapper at `vit_tta.py` using **SAE-Tester**
- Simple evaluation logic at `evalate.py`
- Additional experiments coming up...


## 📜 License & Credits

### Reference Implementations

- [SAE for ViT](https://github.com/HugoFry/mats_sae_training_for_ViTs)
- [SAELens](https://github.com/jbloomAus/SAELens)
- [Differentiable and Fast Geometric Median in NumPy and PyTorch](https://github.com/krishnap25/geom_median)
- [Self-regulating Prompts: Foundational Model Adaptation without Forgetting [ICCV 2023]](https://github.com/muzairkhattak/PromptSRC)
  - Used in: `configs/` and `msrc/models/`
- [MaPLe: Multi-modal Prompt Learning CVPR 2023](https://github.com/muzairkhattak/multimodal-prompt-learning)
  - Used in: `configs/models/maple/...yaml` and `data/clip/maple/imagenet/model.pth.tar-2`

### License Notice

Our code is distributed under an MIT license, please see the [LICENSE](LICENSE) file for details.
The [NOTICE](NOTICE) file lists license for all third-party code included in this repository.
Please include the contents of the LICENSE and NOTICE files in all re-distributions of this code.

---

### Citation

If you find our code or models useful in your work, please cite our [paper](https://arxiv.org/abs/2412.05276):

```
@inproceedings{
  lim2025patchsae,
  title={Sparse autoencoders reveal selective remapping of visual concepts during adaptation},
  author={Hyesu Lim and Jinho Choi and Jaegul Choo and Steffen Schneider},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=imT03YXlG2}
}
```
