# ParetoPrompt: Multi-Objective Prompt Optimization

**Official Implementation of [ParetoPrompt: Multi-Objective Prompt Optimization](https://openreview.net/pdf?id=HGCk5aaSvE), ICLR 2025**

Guang Zhao¹, Byung-Jun Yoon¹², Gilchan Park¹, Shantenu Jha³⁴⁵, Shinjae Yoo¹, Xiaoning Qian¹²  
¹Brookhaven National Laboratory ²Texas A&M University ³Princeton Plasma Physics Laboratory  
⁴Rutgers University – New Brunswick ⁵Princeton University

---

## Overview

ParetoPrompt introduces a reinforcement learning framework for multi-objective prompt optimization.  
Unlike traditional single-objective methods that rely on scalarized rewards, ParetoPrompt directly models Pareto dominance between prompts to efficiently approximate the Pareto front — capturing diverse trade-offs among competing goals such as style, fluency, conciseness, and factuality.

This approach removes the need for hand-tuned weights or scalarization functions, providing a flexible, interpretable, and efficient framework for aligning prompts with multi-dimensional objectives in large language models (LLMs).

---

## Key Features

- Preference-based RL formulation for prompt optimization  
- Pareto-dominance-driven reward modeling without scalarization  
- Exploration of the full Pareto front for diverse trade-offs  
- Cross-metric generalization between training and evaluation criteria  
- Extensible API supporting GPT-2, DistilGPT-2, and custom objective modules  

---

## Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/guangzhao27/ParetoPrompt.git
cd ParetoPrompt
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

(Optional) For reproducibility using Conda:
```bash
conda create -n paretoprompt python=3.10
conda activate paretoprompt
pip install -r requirements.txt
```

### 3. Download pretrained classifier and dataset

#### Style classifier
Download the pretrained Yelp BERT style classifier from Google Drive:  
https://drive.google.com/file/d/1AUBbpFcfBkKh5WUGwdXFhHxzZspPRn7W/view?usp=sharing

Unzip and place the model under:
```
ParetoPrompt/style_classifiers/
```

#### Text-style transfer dataset
Download the Yelp dataset used for text-style transfer from RLPrompt examples:  
https://github.com/mingkaid/rl-prompt/tree/main/examples/text-style-transfer/data/yelp

You can also fetch it using:
```bash
mkdir -p data/yelp
cd data/yelp
wget https://raw.githubusercontent.com/mingkaid/rl-prompt/main/examples/text-style-transfer/data/yelp/train.txt
wget https://raw.githubusercontent.com/mingkaid/rl-prompt/main/examples/text-style-transfer/data/yelp/test.txt
wget https://raw.githubusercontent.com/mingkaid/rl-prompt/main/examples/text-style-transfer/data/yelp/val.txt
cd ../..
```

---

## Example Usage

Run text-style transfer training:
```bash
bash ./examples/text-style-transfer/scripts/run-tst.sh
```

---

## Citation

If you use this work, please cite:

```
@inproceedings{zhao2025paretoprompt,
  title={ParetoPrompt: Multi-Objective Prompt Optimization},
  author={Zhao, Guang and Yoon, Byung-Jun and Park, Gilchan and Jha, Shantenu and Yoo, Shinjae and Qian, Xiaoning},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

---

## License and Acknowledgment

This repository is released under the MIT License.

It includes limited code components adapted from the open-source RLPrompt
project (MIT License, © 2022 Mingkai Deng et al.), which served as an implementation reference for early reinforcement-learning modules.
