<div align="center">

<!-- <!-- <h1 style="color: #33A6B8; font-family: Helvetica"> OmniLMM </h1> -->

<img src="./assets/title-2.png" width="200em" ></img>

**Large multi-modal models for strong performance and efficient deployment**
<p align="center">
  OmniLMM-3B  <a href="https://huggingface.co/openbmb/MiniCPM-V/">🤗</a> <a href="http://120.92.209.146:80/">🤖</a> |
  OmniLMM-12B <a href="https://huggingface.co/openbmb/OmniLMM-12B/">🤗</a> <a href="http://120.92.209.146:8081">🤖</a>
</p>

</div>


**OmniLMM** is a family of open-source large multimodal models (LMMs) adept at vision & language modeling. The model processes images and text inputs and delivers high-quality text outputs. We release two featured versions of OmniLMM that are targeted at **strong performance and efficient deployment**:

- **OmniLMM-12B**: Leading performance among comparable-sized models on multiple benchmarks.

- **OmniLMM-3B**: Frontier end device multi-modal conversation with promising performance.


[中文文档](./README_zh.md)

## Contents
- [Contents](#contents)
- [OmniLMM-12B](#omnilmm-12b)
  - [Evaluation](#evaluation)
  - [Examples](#examples)
- [OmniLMM-3B](#omnilmm-3b)
  - [Evaluation](#evaluation-1)
  - [Examples](#examples-1)
- [Demo](#demo)
- [Install](#install)
- [Inference](#inference)
  - [Model Zoo](#model-zoo)
  - [Multi-turn Conversation](#multi-turn-conversation)
  - [Inference on Mac](#inference-on-mac)
  - [Deployment on Mobile Phone](#deployment-on-mobile-phone)
- [TODO](#todo)
- [Model License](#model-license)
- [Statement](#statement)
- [Institutions](#institutions)

## OmniLMM-12B
**OmniLMM-12B** is the most capable version. The model is built based on EVA02-5B and Zephyr-7B-β, connected with a perceiver resampler layer, and trained on multimodal data in a curriculum fashion. The model has three notable features:

- 🔥 **Strong Performance.**

  OmniLMM-12B achieves **leading performance** among models with comparable sizes, surpassing established LMMs on multiple benchmarks (including MME, MMBench, SEED-Bench, etc). The model also endows rich multi-modal world knowledge.

- 🏆 **Trustworthy Behavior.**

  LMMs are known for suffering from hallucination, often generating text that is not factually grounded in images (e.g., faithfully describing non-existing objects in images). OmniLMM-12B is **the first state-of-the-art open-source LMM aligned via multimodal RLHF for trustworthy behavior** (using the recent [RLHF-V](https://rlhf-v.github.io/) technique). It **ranks #1** among open-source models on [MMHal-Bench](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench), and **outperforms GPT-4V** on [Object HalBench](https://arxiv.org/abs/2312.00849).

- 🕹 **Real-time Multimodal Interaction.**

  We combine the OmniLMM-12B and GPT-3.5 (text-only) into a **real-time multimodal interactive assistant**. The assistant accepts video streams from the camera and speech streams from the microphone and emits speech output. While still primary, we find the model can **replicate some of the fun cases shown in the Gemini Demo video, without any video edition**.


### Evaluation
<div align="center">
    <img src=assets/eval_radar.png width=66% />
</div>
<details>
<summary>Click to view results on MME, MMBench, MMMU, MMBench, MMHal-Bench, Object HalBench, SeedBench, LLaVA Bench W, MathVista. </summary>

<table>
<thead>
  <tr>
    <th align="left">Model</th>
    <th>Size</th>
    <th>MME</th>
    <th nowrap="nowrap">MMB dev (en)</th>
    <th nowrap="nowrap" >MMMU val</th>
    <th nowrap="nowrap" >MMHal-Bench</th>
    <th nowrap="nowrap" >Object HalBench</th>
    <th nowrap="nowrap" >SeedBench-I</th>
    <th>MathVista</th>
    <th nowrap="nowrap" >LLaVA Bench W</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td align="left">GPT-4V†</td>
    <td>-</td>
    <td>1409</td>
    <td>75.1 </td>
    <td>56.8</td>
    <td>3.53 / 70.8</td>
    <td>86.4 / 92.7</td>
    <td>71.6 </td>
    <td>47.8 </td>
    <td>93.1 </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left">Qwen-VL-Plus†</td>
    <td>-</td>
    <td>1681</td>
    <td>66.2 </td>
    <td>45.2</td>
    <td>- </td>
    <td>- </td>
    <td>65.7 </td>
    <td>36.0 </td>
    <td>73.7 </td>
  </tr>
  <tr>
    <td align="left">Yi-VL 6B</td>
    <td align="right">6.7B </td>
    <td>- </td>
    <td>68.2 </td>
    <td>39.1 </td>
    <td>- </td>
    <td>- </td>
    <td>66.1 </td>
    <td>28.0 </td>
    <td>39.9 </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left" >Qwen-VL-Chat</td>
    <td align="right">9.6B</td>
    <td>1488</td>
    <td>60.6 </td>
    <td>35.9</td>
    <td>2.93 / 59.4</td>
    <td>56.2 / 80.0</td>
    <td>64.8 </td>
    <td>33.8 </td>
    <td>67.7 </td>
  </tr>
  <tr>
    <td align="left" >CogVLM</td>
    <td align="right">17.4B</td>
    <td>1438</td>
    <td>63.7 </td>
    <td>32.1 </td>
    <td>2.68 / 52.1 </td>
    <td>73.6 / 87.4 </td>
    <td>68.8 </td>
    <td>34.7 </td>
    <td>73.9 </td>
  </tr>
  <tr>
    <td align="left" >LLaVA 1.5</td>
    <td align="right">13.6B </td>
    <td>1531 </td>
    <td>68.2 </td>
    <td>36.4 </td>
    <td>2.71 / 51.0 </td>
    <td>53.7 / 77.4 </td>
    <td>68.1 </td>
    <td>26.4 </td>
    <td>64.6 </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left" ><b>OmniLMM-12B</b></td>
    <td align="right">11.6B </td>
    <td>1637 </td>
    <td>71.6 </td>
    <td>40.7 </td>
    <td>3.45 / 68.8 </td>
    <td>90.3 / 95.5 </td>
    <td>71.1 </td>
    <td>34.9 </td>
    <td>72.0 </td>
  </tr>
</tbody>
</table>
<small>†: Proprietary models</small>
<br>
</details>

### Examples

<table align="center" >
  <p align="center" >
    <img src="assets/omnilmm-12b-examples_2.png" />
  </p>
</table>


We combine the OmniLMM-12B and GPT-3.5 (text-only) into a **real-time multimodal interactive assistant**. Video frames are described in text using OmniLMM-12B, and ChatGPT 3.5 (text-only) is employed to generate response according to the descriptions and user prompts. The demo video is a raw recording without edition.

<div align="center" >
  <video controls src="https://github.com/OpenBMB/OmniLMM/assets/157115220/485a8f52-fb4d-4eca-8fee-506347efcfc6" type="video/mp4" width=80%/>
</div>

## OmniLMM-3B
**OmniLMM-3B** (i.e., MiniCPM-V) is an efficient version with promising performance for deployment. The model is built based on SigLip-400M and [MiniCPM-2.4B](https://github.com/OpenBMB/MiniCPM/), connected by a perceiver resampler. Notable features of OmniLMM-3B include:

- ⚡️ **High Efficiency.**

  OmniLMM-3B can be **efficiently deployed on most GPU cards and personal computers**, and **even on end devices such as mobile phones**. In terms of visual encoding, we compress the image representations into 64 tokens via a perceiver resampler, which is significantly fewer than other LMMs based on MLP architecture (typically > 512 tokens). This allows OmniLMM-3B to operate with **much less memory cost and higher speed during inference**.

- 🔥 **Promising Performance.**

  OmniLMM-3B achieves **state-of-the-art performance** on multiple benchmarks (including MMMU, MME, and MMbech, etc) among models with comparable sizes, surpassing existing LMMs built on Phi-2. It even **achieves comparable or better performance than the 9.6B Qwen-VL-Chat**.

- 🙌 **Bilingual Support.**

  OmniLMM-3B is **the first end-deployable LMM supporting bilingual multimodal interaction in English and Chinese**. This is achieved by generalizing multimodal capabilities across languages, a technique from the ICLR 2024 spotlight [paper](https://arxiv.org/abs/2308.12038).

### Evaluation

<div align="center">

<table style="margin: 0px auto;">
<thead>
  <tr>
    <th align="left">Model</th>
    <th>Size</th>
    <th nowrap="nowrap" >Visual Tokens</th>
    <th>MME</th>
    <th nowrap="nowrap" >MMB dev (en)</th>
    <th nowrap="nowrap" >MMB dev (zh)</th>
    <th nowrap="nowrap" >MMMU val</th>
    <th nowrap="nowrap" >CMMMU val</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td align="left">LLaVA-Phi</td>
    <td align="right">3B</td>
    <td>576</td>
    <td>1335</td>
    <td>59.8</td>
    <td>- </td>
    <td>- </td>
    <td>- </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left">MobileVLM</td>
    <td align="right">3B</td>
    <td>144</td>
    <td>1289</td>
    <td>59.6</td>
    <td>- </td>
    <td>- </td>
    <td>- </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left" >Imp-v1</td>
    <td align="right">3B</td>
    <td>576</td>
    <td>1434</td>
    <td>66.5</td>
    <td>- </td>
    <td>- </td>
    <td>- </td>
  </tr>
  <tr>
    <td  nowrap="nowrap" align="left" >Qwen-VL-Chat</td>
    <td align="right" >9.6B</td>
    <td>256</td>
    <td>1487</td>
    <td>60.6 </td>
    <td>56.7 </td>
    <td>35.9 </td>
    <td>30.7 </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left" >CogVLM</td>
    <td align="right">17.4B </td>
    <td>1225</td>
    <td>1438 </td>
    <td>63.7 </td>
    <td>53.8 </td>
    <td>32.1 </td>
    <td>- </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left" ><b>OmniLMM-3B</b></td>
    <td align="right">3B </td>
    <td>64</td>
    <td>1452 </td>
    <td>67.9 </td>
    <td>65.3 </td>
    <td>37.2 </td>
    <td>32.1 </td>
  </tr>
</tbody>
</table>

</div>

### Examples

We deploy OmniLLM-3B on end devices. The demo video is the raw screen recording on a OnePlus 9R without edition.

<table align="center">
    <p align="center">
      <img src="assets/gif_cases/蛇_cn.gif" width=36%/>
      <img src="assets/gif_cases/Mushroom_en.gif" width=36%/>
    </p>
</table>

## Demo
Click here to try out the Demo of [OmniLMM-12B](http://120.92.209.146:8081) and [OmniLMM-3B](http://120.92.209.146:80).

## Install

1. Clone this repository and navigate to the source folder

```bash
git clone https://github.com/OpenBMB/OmniLMM.git
cd OmniLMM
```

2. Create conda environment

```Shell
conda create -n OmniLMM python=3.10 -y
conda activate OmniLMM
```

3. Install dependencies

```shell
pip install -r requirements.txt
```

## Inference

### Model Zoo
| Model                | Description       | Download Link |
|:----------------------|:-------------------|:---------------:|
| OmniLMM-12B | The most capable version with strong performance.   |  [🤗](https://huggingface.co/openbmb/OmniLMM-12B) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/OmniLMM-12B/files) |
| OmniLMM-3B  | The efficient version for end device deployment.    |  [🤗](https://huggingface.co/openbmb/MiniCPM-V) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V/files) |


### Multi-turn Conversation
Please refer to the following codes to run `OmniLMM`.

<div align="center">
<img src="assets/worldmap_ck.jpg" width="500px">
</div>


```python
from chat import OmniLMMChat, img2base64

chat_model = OmniLMMChat('openbmb/OmniLMM-12B') # or 'openbmb/MiniCPM-V'

im_64 = img2base64('./assets/worldmap_ck.jpg')

# First round chat
msgs = [{"role": "user", "content": "What is interesting about this image?"}]

inputs = {"image": im_64, "question": json.dumps(msgs)}
answer = chat_model.chat(inputs)
print(answer)

# Second round chat
# pass history context of multi-turn conversation
msgs.append({"role": "assistant", "content": answer})
msgs.append({"role": "user", "content": "Where is China in the image"})

inputs = {"image": im_64, "question": json.dumps(msgs)}
answer = chat_model.chat(inputs)
print(answer)
```

We can obtain the following results:
```
"The interesting aspect of this image is the shape of the chicken nuggets on the pan. The nuggets are shaped like the continents of the world, which is an unusual and creative way to present the food. It adds a fun and playful element to the meal, making it more visually appealing and engaging."

"In the image, China is located on the right side of the pan. It is one of the nuggets shaped like the continents of the world, and its placement on the right side of the pan is consistent with its geographical location in the real world"
```

### Inference on Mac
<details>
<summary>Click to view example, OmniLMM-3B (i.e., MiniCPM-V) can run on Mac with MPS (Apple silicon or AMD GPUs). </summary>

```python
# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to(device='mps', dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True)
model.eval()

image = Image.open('./assets/worldmap_ck.jpg').convert('RGB')
question = 'What is interesting about this image?'
msgs = [{'role': 'user', 'content': question}]

answer, context, _ = model.chat(
    image=image,
    msgs=msgs,
    context=None,
    tokenizer=tokenizer,
    sampling=True
)
print(answer)
```
Run with command:
```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python test.py
```
</details>

### Deployment on Mobile Phone
Currently OmniLMM-3B (i.e., MiniCPM-V) can be deployed on mobile phones with Android and Harmony operating systems. 🚀 Try it out [here](https://github.com/OpenBMB/mlc-MiniCPM).

## TODO

- [ ] Fine-tuning support
- [ ] Local Web-UI deployment
- [ ] Code release for real-time interactive assistant

## Model License

The code in this repo is released according to [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE)

The usage of OmniLMMs' parameters is subject to "[General Model License Agreement - Source Notes - Publicity Restrictions - Commercial License](https://github.com/OpenBMB/General-Model-License/blob/main/通用模型许可协议-来源说明-宣传限制-商业授权.md)"

The parameters are fully open to acedemic research

Please contact cpm@modelbest.cn to obtain a written authorization for commercial uses. Free commercial use is also allowed after registration.

## Statement

As LMMs, OmniLMMs generate contents by learning a large mount of multimodal corpora, but it cannot comprehend, express personal opinions or make value judgement. Anything generated by OmniLMMs does not represent the views and positions of the model developers

We will not be liable for any problems arising from the use of OmniLMM open source models, including but not limited to data security issues, risk of public opinion, or any risks and problems arising from the misdirection, misuse, dissemination or misuse of the model.


## Institutions

This project is developed by the following institutions:

- <img src="assets/thunlp.png" width="28px"> [THUNLP](https://nlp.csai.tsinghua.edu.cn/)
- <img src="assets/modelbest.png" width="28px"> [ModelBest](https://modelbest.cn/)
- <img src="assets/zhihu.webp" width="28px"> [Zhihu](https://www.zhihu.com/ )

