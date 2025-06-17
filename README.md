# Qwen2.5-Omni
<p align="center">
        💜 <a href="https://chat.qwenlm.ai/"><b>Qwen Chat</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/collections/Qwen/qwen25-omni-67de1e5f0f9464dc6314b36e">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/collections/Qwen25-Omni-a2505ce0d5514e">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://qwenlm.github.io/blog/qwen2.5-omni/">Blog</a>&nbsp&nbsp | &nbsp&nbsp📚 <a href="https://github.com/QwenLM/Qwen2.5-Omni/tree/main/cookbooks">Cookbooks</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2503.20215">Paper</a>&nbsp&nbsp
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen2.5-Omni-7B-Demo ">Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://help.aliyun.com/zh/model-studio/user-guide/qwen-omni">API</a>
<!-- &nbsp&nbsp | &nbsp&nbsp🖥️ <a href="https://gallery.pai-ml.com/#/preview/deepLearning/cv/qwen2.5-vl">PAI-DSW</a> -->
</p>


## Contents <!-- omit in toc -->

- [Overview](#overview)
  - [Introduction](#introduction)
  - [Key Features](#key-features)
  - [Model Architecture](#model-architecture)
  - [Performance](#performance)
- [Quickstart](#quickstart)
  - [Transformers Usage](#--transformers-usage)
  - [ModelScope Usage](#-modelscope-usage)
  - [GPTQ-Int4 and AWQ Usage](#gptq-int4-and-awq-usage)
  - [Usage Tips](#usage-tips)
  - [Cookbooks for More Usage Cases](#cookbooks-for-more-usage-cases)
  - [API inference](#api-inference)
  - [Customization Settings](#customization-settings)
- [Chat with Qwen2.5-Omni](#chat-with-qwen25-omni)
  - [Online Demo](#online-demo)
  - [Launch Local Web UI Demo](#launch-local-web-ui-demo)
  - [Real-Time Interaction](#real-time-interaction)
- [Deployment with vLLM](#deployment-with-vllm)
- [Deployment with MNN](#deployment-with-mnn)
- [Docker](#-docker)
<!-- - [Citation](#citation) -->



## Quickstart

Below, we provide simple examples to show how to use Qwen2.5-Omni with 🤖 ModelScope and 🤗 Transformers.

The codes of Qwen2.5-Omni has been in the latest Hugging face transformers and we advise you to install with command:
```
pip install transformers==4.52.3
pip install accelerate
```
or you might encounter the following error:
```
KeyError: 'qwen2_5_omni'
```
and you can also use our [official docker image](#-docker) to start without building from source.

We offer a toolkit to help you handle various types of audio and visual input more conveniently, as if you were using an API. This includes base64, URLs, and interleaved audio, images and videos. You can install it using the following command and make sure your system has `ffmpeg` installed:

```bash
# It's highly recommended to use `[decord]` feature for faster video loading.
pip install qwen-omni-utils[decord] -U
```

If you are not using Linux, you might not be able to install `decord` from PyPI. In that case, you can use `pip install qwen-omni-utils -U` which will fall back to using torchvision for video processing. However, you can still [install decord from source](https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source) to get decord used when loading video.

We are preparing [cookbooks](https://github.com/QwenLM/Qwen2.5-Omni/tree/main/cookbooks) for many capabilities, including audio understanding, voice chatting, screen recording interaction, video information extracting, omni chatting and more. Welcome to learn more!

### 🤗  Transformers Usage

Here we show a code snippet to show you how to use the chat model with `transformers` and `qwen_omni_utils`:

```python
import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# default: Load the model on the available device(s)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto")

# We recommend enabling flash_attention_2 for better acceleration and memory saving.
# model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-Omni-7B",
#     torch_dtype="auto",
#     device_map="auto",
#     attn_implementation="flash_attention_2",
# )

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"},
        ],
    },
]

# set use audio in video
USE_AUDIO_IN_VIDEO = True

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

# Inference: Generation of the output text and audio
text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)
sf.write(
    "output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)
```

#### Minimum GPU memory requirements

| Model | Precision | 15(s) Video | 30(s) Video | 60(s) Video |
|--------------|-----------| ------------- | ------------- | ------------------ |
| Qwen-Omni-3B | FP32      | 89.10 GB      | Not Recommend | Not Recommend      |
| Qwen-Omni-3B | BF16      | 18.38 GB      | 22.43 GB      | 28.22 GB           |
| Qwen-Omni-7B | FP32      | 93.56 GB      | Not Recommend | Not Recommend      |
| Qwen-Omni-7B | BF16      | 31.11 GB      | 41.85 GB      | 60.19 GB           |

Note: The table above presents the theoretical minimum memory requirements for inference with `transformers` and `BF16` is test with `attn_implementation="flash_attention_2"`. However, in practice, the actual memory usage is typically at least 1.2 times higher. For more information, see the linked resource [here](https://huggingface.co/docs/accelerate/main/en/usage_guides/model_size_estimator). We are currently planning to develop a version that can perform inference with lower resource consumption requirements so that Qwen2.5-Omni can run on most platforms. Stay tuned!

<details>
<summary>Video URL resource usage</summary>

Video URL compatibility largely depends on the third-party library version. The details are in the table below. Change the backend by `FORCE_QWENVL_VIDEO_READER=torchvision` or `FORCE_QWENVL_VIDEO_READER=decord` if you prefer not to use the default one.

| Backend     | HTTP | HTTPS |
|-------------|------|-------|
| torchvision >= 0.19.0 | ✅  | ✅   |
| torchvision < 0.19.0  | ❌  | ❌   |
| decord      | ✅  | ❌   |
</details>

<details>
<summary>Batch inference</summary>

The model can batch inputs composed of mixed samples of various types such as text, images, audio and videos as input when `return_audio=False` is set. Here is an example.

```python
# Sample messages for batch inference

# Conversation with video only
conversation1 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "/path/to/video.mp4"},
        ]
    }
]

# Conversation with audio only
conversation2 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": "/path/to/audio.wav"},
        ]
    }
]

# Conversation with pure text
conversation3 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": "who are you?"
    }
]


# Conversation with mixed media
conversation4 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/path/to/image.jpg"},
            {"type": "video", "video": "/path/to/video.mp4"},
            {"type": "audio", "audio": "/path/to/audio.wav"},
            {"type": "text", "text": "What are the elements can you see and hear in these medias?"},
        ],
    }
]

# Combine messages for batch processing
conversations = [conversation1, conversation2, conversation3, conversation4]

# set use audio in video
USE_AUDIO_IN_VIDEO = True

# Preparation for batch inference
text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO)

inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

# Batch Inference
text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)
text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)
```
</details>


### 🤖 ModelScope Usage
We strongly advise users especially those in mainland China to use ModelScope, `snapshot_download` can help you solve issues concerning downloading checkpoints.

### GPTQ-Int4 and AWQ Usage

To improve the Qwen2.5-Omni-7B's operability on devices with constrained GPU memory, we implemented 4-bit quantization of the Thinker's weights using GPTQ and AWQ, effectively reducing GPU VRAM usage. Ohter key optimizations include:
* Enhanced the inference pipeline to load model weights on-demand for each module and offload them to CPU memory once inference is complete, preventing peak VRAM usage from becoming excessive.
* Converted the code2wav module to support streaming inference, thereby avoiding the pre-allocation of excessive GPU memory.
* Adjusted the ODE solver from a second-order (RK4) to a first-order (Euler) method to further decrease computational overhead.

These improvements aim to ensure efficient performance of Qwen2.5-Omni across a range of hardware configurations, particularly those with lower GPU memory availability (RTX3080, 4080, 5070, etc). Currently, the relevant models and usage methods can be obtained from Hugging Face ([GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-Omni-7B-GPTQ-Int4)|[AWQ](https://huggingface.co/Qwen/Qwen2.5-Omni-7B-AWQ)) and ModelScope ([GPTQ-Int4](https://modelscope.cn/models/Qwen/Qwen2.5-Omni-7B-GPTQ-Int4)|[AWQ](https://modelscope.cn/models/Qwen/Qwen2.5-Omni-7B-AWQ)). As below, we provide simple example to show how to use Qwen2.5-Omni-7B-GPTQ-Int4 with `gptqmodel` as follows:
```
pip install transformers==4.52.3
pip install accelerate
pip install gptqmodel==2.0.0
pip install numpy==2.0.0

git clone https://github.com/QwenLM/Qwen2.5-Omni.git

cd Qwen2.5-Omni/low-VRAM-mode/

CUDA_VISIBLE_DEVICES=0 python3 low_VRAM_demo_gptq.py
```

To use Qwen2.5-Omni-7B-AWQ with `autoawq` please run:
```
pip install transformers==4.52.3
pip install accelerate
pip install autoawq==0.2.9

git clone https://github.com/QwenLM/Qwen2.5-Omni.git

cd Qwen2.5-Omni/low-VRAM-mode/

CUDA_VISIBLE_DEVICES=0 python3 low_VRAM_demo_awq.py
```

The following two tables present a performance comparison and GPU memory consumption between Qwen2.5-Omni-7B-GPTQ-Int4/Qwen2.5-Omni-7B-AWQ and Qwen2.5-Omni-7B on specific evaluation benchmarks. The data demonstrates that the GPTQ-Int4/AWQ model maintains comparable performance while reducing GPU memory requirements by over 50%+, enabling a broader range of devices to run and experience the high-performance Qwen2.5-Omni-7B model. Notably, the GPTQ-Int4/AWQ variant exhibits slightly slower inference speeds compared to the native Qwen2.5-Omni-7B model due to quantization techniques and CPU offload mechanisms.

| Evaluation Set | Task | Metrics | Qwen2.5-Omni-7B | Qwen2.5-Omni-7B-GPTQ-Int4 | Qwen2.5-Omni-7B-AWQ |
|--------------|-----------| ------------- | ------------- | ------------------ |  ------------------ |
| LibriSpeech test-other   | ASR                   | WER ⬇️      | 3.4   | 3.71  | 3.91  |
| WenetSpeech test-net     | ASR                   | WER ⬇️      | 5.9   | 6.62  | 6.31  |
| Seed-TTS test-hard       | TTS (Speaker: Chelsie)| WER ⬇️      | 8.7   | 10.3  | 8.88  |
| MMLU-Pro                 | Text -> Text          | Accuracy ⬆️ | 47.0  | 43.76 | 45.66 |
| OmniBench                | Speech -> Text        | Accuracy ⬆️ | 56.13 | 53.59 | 54.64 |
| VideoMME                 | Multimodality -> Text | Accuracy ⬆️ | 72.4  | 68.0  | 72.0  |

|Model | Precision | 15(s) Video | 30(s) Video | 60(s) Video |
|--------------|-----------| ------------- | ------------- | ------------------ |
| Qwen-Omni-7B | FP32      | 93.56 GB      | Not Recommend | Not Recommend      |
| Qwen-Omni-7B | BF16      | 31.11 GB      | 41.85 GB      | 60.19 GB           |
| Qwen-Omni-7B | GPTQ-Int4 | 11.64 GB      | 17.43 GB      | 29.51 GB           |
| Qwen-Omni-7B | AWQ       | 11.77 GB      | 17.84 GB      | 30.31 GB           |

### Usage Tips

#### Prompt for audio output
If users need audio output, the system prompt must be set as "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.", otherwise the audio output may not work as expected.
```
{
    "role": "system",
    "content": [
          {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
    ]
}
```

#### Use audio in video
In the process of multimodal interaction, the videos provided by users are often accompanied by audio (such as questions about the content in the video, or sounds generated by certain events in the video). This information is conducive to the model providing a better interactive experience. So we provide the following options for users to decide whether to use audio in video.
```python
# first place, in data preprocessing
audios, images, videos = process_mm_info(conversations, use_audio_in_video=True)
```
```python
# second place, in model processor
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", 
                   padding=True, use_audio_in_video=True)
```
```python
#  third place, in model inference
text_ids, audio = model.generate(**inputs, use_audio_in_video=True)
```
It is worth noting that during a multi-round conversation, the `use_audio_in_video` parameter in these places must be set to the same, otherwise unexpected results will occur.

#### Use audio output or not

The model supports both text and audio outputs, if users do not need audio outputs, they can call `model.disable_talker()` after init the model. This option will save about `2GB` of GPU memory but the `return_audio` option for `generate` function will only allow to be set at `False`.
```python
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype="auto",
    device_map="auto"
)
model.disable_talker()
```

In order to obtain a flexible experience, we recommend that users can decide whether to return audio when `generate` function is called. If `return_audio` is set to `False`, the model will only return text outputs to get text responses faster.

```python
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype="auto",
    device_map="auto"
)
...
text_ids = model.generate(**inputs, return_audio=False)
```

#### Change voice type of output audio
Qwen2.5-Omni supports the ability to change the voice of the output audio. The `"Qwen/Qwen2.5-Omni-7B"` checkpoint supports two voice types as follows:

| Voice Type | Gender | Description |
|------------|--------|-------------|
| Chelsie    | Female | A honeyed, velvety voice that carries a gentle warmth and luminous clarity.|
| Ethan      | Male   | A bright, upbeat voice with infectious energy and a warm, approachable vibe.|

Users can use the `speaker` parameter of `generate` function to specify the voice type. By defalut, if `speaker` is not specified, the default voice type is `Chelsie`.

```python
text_ids, audio = model.generate(**inputs, speaker="Chelsie")
```

```python
text_ids, audio = model.generate(**inputs, speaker="Ethan")
```

#### Flash-Attention 2 to speed up generation

First, make sure to install the latest version of Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

Also, you should have hardware that is compatible with FlashAttention 2. Read more about it in the official documentation of the [flash attention repository](https://github.com/Dao-AILab/flash-attention). FlashAttention-2 can only be used when a model is loaded in `torch.float16` or `torch.bfloat16`.

To load and run a model using FlashAttention-2, add `attn_implementation="flash_attention_2"` when loading the model:

```python
from transformers import Qwen2_5OmniForConditionalGeneration

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```


### Cookbooks for More Usage Cases 

| Cookbook | Description | Open |
| -------- | ----------- | ---- |
| [Universal Audio Understanding](https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/universal_audio_understanding.ipynb) | Speech recongnition, speech-to-text translation and audio analysis. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/universal_audio_understanding.ipynb) |
 | [Voice Chatting](https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/voice_chatting.ipynb) | Chatting with Qwen2.5-Omni by voice input and output. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/voice_chatting.ipynb) |
 | [Screen Recording Interaction](https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/screen_recording_interaction.ipynb) | Get the information and content you want to know by asking questions in real time on the recording screen. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/screen_recording_interaction.ipynb) |
 | [Video Information Extracting](https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/video_information_extracting.ipynb) | Obtaining information from the video stream. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/video_information_extracting.ipynb) |
 | [Omni Chatting for Music](https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/omni_chatting_for_music.ipynb) | Chat with Qwen2.5-Omni about music content in a audio and video stream. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/omni_chatting_for_music.ipynb) |
 | [Omni Chatting for Math](https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/omni_chatting_for_math.ipynb) | Chat with Qwen2.5-Omni about math content in a audio and video stream. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/omni_chatting_for_math.ipynb) |
 | [Multi Round Omni Chatting](https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/multi_round_omni_chatting.ipynb) | Conducted multiple rounds of audio and video dialogues with Qwen2.5-Omni to provide the most comprehensive ability demonstration. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/multi_round_omni_chatting.ipynb) |

### API Inference

To explore Qwen2.5-Omni, we encourage you to test our cutting-edge API service for a faster and efficient experience.

#### Installation
```bash
pip install openai
```

#### Examples
You can use the OpenAI API service to interact with Qwen2.5-Omni like below. And for more usage, please refer to the tutorial at [aliyun](https://help.aliyun.com/zh/model-studio/user-guide/qwen-omni).
```python
import base64
import numpy as np
import soundfile as sf

from openai import OpenAI

client = OpenAI(
    api_key="your_api_key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

messages = [
    {
        "role": "system",
        "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
    },
    {
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"},
        ],
    },
]

# Qwen-Omni only supports stream mode
completion = client.chat.completions.create(
    model="qwen-omni-turbo",
    messages=messages,
    modalities=["text", "audio"],
    audio={
        "voice": "Cherry", # Cherry, Ethan, Serena, Chelsie is available
        "format": "wav"
    },
    stream=True,
    stream_options={"include_usage": True}
)

text = []
audio_string = ""
for chunk in completion:
    if chunk.choices:
        if hasattr(chunk.choices[0].delta, "audio"):
            try:
                audio_string += chunk.choices[0].delta.audio["data"]
            except Exception as e:
                text.append(chunk.choices[0].delta.audio["transcript"])
    else:
        print(chunk.usage)

print("".join(text))
wav_bytes = base64.b64decode(audio_string)
wav_array = np.frombuffer(wav_bytes, dtype=np.int16)
sf.write("output.wav", wav_array, samplerate=24000)
```
### Customization Settings

Since Qwen2.5-Omni does not support prompt settings when using [audio output](#prompt-for-audio-output) (including local deployment and API inference), we suggest that if you need to control the output of the model or modify the personality settings of the model, you can try adding similar content to the conversation template as follows:

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "You are a shopping guide, now responsible for introducing various products."},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Sure, I got it."},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Who are you?"},
        ],
    },
]
```

## Chat with Qwen2.5-Omni

### Online Demo
Without deployment, you can experience online web demo directly by visiting our [Hugginface Spaces](https://huggingface.co/spaces/Qwen/Qwen2.5-Omni-7B-Demo) and [Modelscope Studio](https://modelscope.cn/studios/Qwen/Qwen2.5-Omni-Demo).

### Launch Local Web UI Demo

In this section, we provide instructions for users to build a web-based user interface (UI) demo. This UI demo allows users to interact with a predefined model or application through a web browser. Follow the steps below to get started or you can launch the web demo directly from our [official docker image](#-docker).

#### Installation

Before you begin, ensure that you have the required dependencies installed on your system. You can install them by running the following command:

```bash
pip install -r requirements_web_demo.txt
```

#### Running the Demo with FlashAttention-2

Once the required packages are installed, you can launch the web demo using the following command. This command will start a web server and provide you with a link to access the UI in your web browser.

**Recommended**: For enhanced performance and efficiency, especially in multi-image and video processing scenarios, we strongly recommend using [FlashAttention-2](https://github.com/Dao-AILab/flash-attention). FlashAttention-2 provides significant improvements in memory usage and speed, making it ideal for handling large-scale models and data processing.

To enable FlashAttention-2, use the following command:

```bash
# default for Qwen2.5-Omni-7B
python web_demo.py --flash-attn2
```
```bash
# for Qwen2.5-Omni-3B
python web_demo.py --flash-attn2 -c Qwen/Qwen2.5-Omni-3B
```

This will load the model with FlashAttention-2 enabled.

**Default Usage**: If you prefer to run the demo without FlashAttention-2 or if you do not specify the `--flash-attn2` option, the demo will load the model using the standard attention implementation:

```bash
# default for Qwen2.5-Omni-7B
python web_demo.py
```
```bash
# for Qwen2.5-Omni-3B
python web_demo.py -c Qwen/Qwen2.5-Omni-3B
```

After running the command, you’ll see a link generated in the terminal similar to this:

```
Running on local: http://127.0.0.1:7860/
```

Copy this link and paste it into your browser to access the web UI, where you can interact with the model by inputting text, uploading audios/images/videos, changing voice type or using any other provided functionalities.


### Real-Time Interaction
The streaming Real-time interaction with Qwen2.5-Omni is available now, please visit [Qwen Chat](https://chat.qwen.ai/) and select the voice/video calls in the chat box to experience. 


## Deployment with vLLM

We recommend using vLLM for fast Qwen2.5-Omni deployment and inference. You need to install from our provided [source](https://github.com/fyabc/vllm/tree/qwen2_omni_public) to get vLLM support for Qwen2.5-Omni or use our [official docker image](#-docker). You can also check [vLLM official documentation](https://docs.vllm.ai/en/latest/serving/multimodal_inputs.html) for more details about online serving and offline inference.

### Installation
```bash
git clone -b qwen2_omni_public https://github.com/fyabc/vllm.git
cd vllm
git checkout de8f43fbe9428b14d31ac5ec45d065cd3e5c3ee0
pip install setuptools_scm torchdiffeq resampy x_transformers qwen-omni-utils accelerate
pip install -r requirements/cuda.txt
pip install --upgrade setuptools wheel
pip install .
pip install transformers==4.52.3
```

### Inference Local

You can use vLLM to inference Qwen2.5-Omni locally, we provide example in [vLLM repo](https://github.com/fyabc/vllm/blob/qwen2_omni_public/examples/offline_inference/qwen2_5_omni/end2end.py) which can generate audio output:

```bash
# git clone -b qwen2_omni_public https://github.com/fyabc/vllm.git
# cd vllm
# git checkout de8f43fbe9428b14d31ac5ec45d065cd3e5c3ee0
# cd examples/offline_inference/qwen2_5_omni/

# only text output for single GPU
python end2end.py --model Qwen/Qwen2.5-Omni-7B --prompt audio-in-video-v2 --enforce-eager --thinker-only

# only text output for multi GPUs (example in 4 GPUs)
python end2end.py --model Qwen/Qwen2.5-Omni-7B --prompt audio-in-video-v2 --enforce-eager --thinker-only --thinker-devices [0,1,2,3] --thinker-gpu-memory-utilization 0.9 

# audio output for single GPU
python end2end.py --model Qwen/Qwen2.5-Omni-7B --prompt audio-in-video-v2 --enforce-eager --do-wave --voice-type Chelsie --warmup-voice-type Chelsie --output-dir output_wav

# audio output for multi GPUs (example in 4 GPUs)
python end2end.py --model Qwen/Qwen2.5-Omni-7B --prompt audio-in-video-v2 --enforce-eager --do-wave --voice-type Chelsie --warmup-voice-type Chelsie --thinker-devices [0,1] --talker-devices [2] --code2wav-devices [3] --thinker-gpu-memory-utilization 0.9 --talker-gpu-memory-utilization 0.9 --output-dir output_wav
```

### vLLM Serve usage
You can also use vLLM serve through `pip install vllm>=0.8.5.post1`, and vLLM serve for Qwen2.5-Omni only supports thinker now, meaning only text output is supported. You can start vLLM servev through the following command:
```bash
# for single GPU
vllm serve /path/to/Qwen2.5-Omni-7B/ --port 8000 --host 127.0.0.1 --dtype bfloat16
# for multi GPUs (example in 4 GPUs)
vllm serve /path/to/Qwen2.5-Omni-7B/ --port 8000 --host 127.0.0.1 --dtype bfloat16 -tp 4
```
Then you can use the chat API as below (via curl for example):
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
        {"type": "audio_url", "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/cough.wav"}},
        {"type": "text", "text": "What is the text in the illustrate ans what it the sound in the audio?"}
    ]}
    ]
    }'
```

## Deployment with MNN

Qwen2.5-Omni is now supported in MNN, enabling deployment on edge devices. The MNN models for Qwen2.5-Omni are available for download through Hugging Face ([7B](https://huggingface.co/taobao-mnn/Qwen2.5-Omni-7B-MNN)|[3B](https://huggingface.co/taobao-mnn/Qwen2.5-Omni-3B-MNN)) and ModelScope ([7B](https://modelscope.cn/models/MNN/Qwen2.5-Omni-7B-MNN)|[3B](https://modelscope.cn/models/MNN/Qwen2.5-Omni-3B-MNN)), along with usage instructions.  For detailed information, you can visit [MNN](https://github.com/alibaba/MNN) to learn about it.

The table below shows memory consumption and inference speed benchmarks for the Qwen2.5-Omni MNN implementation across various mobile SoC platforms.

| Platform | Snapdragon 8 Gen 1 | Snapdragon 8 Elite | Snapdragon 8 Gen 1 | Snapdragon 8 Elite  |
|--------------|-----------| ------------- | ------------- | ------------------ |
| Model Size   | 7B | 7B | 3B | 3B |
| Memory Peak  | 5.8G | 5.8G | 3.6G | 3.6G |
| Thinker Prefill Speed | 25.58 tok/s | 46.32 tok/s | 54.31 tok/s | 55.16 tok/s | 
| Thinker Decode Speed  |  8.35 tok/s | 11.52 tok/s | 15.84 tok/s | 23.31 tok/s | 
| Talker Prefill Speed  | 17.21 tok/s | 97.77 tok/s | 34.58 tok/s | 217.82 tok/s| 
| Talker Decode Speed   | 18.75 tok/s | 38.65 tok/s | 51.90 tok/s | 62.34 tok/s | 
| Code2Wav Speed         |20.83 tok/s | 27.36 tok/s | 28.45 tok/s | 27.36 tok/s | 


## 🐳 Docker

To simplify the deploy process, we provide docker images with pre-build environments: [qwenllm/qwen-omni](https://hub.docker.com/r/qwenllm/qwen-omni). You only need to install the driver and download model files to launch demos.

```bash
docker run --gpus all --ipc=host --network=host --rm --name qwen2.5-omni -it qwenllm/qwen-omni:2.5-cu121 bash
```

And you can also launch the web demo by:
```bash
bash docker/docker_web_demo.sh --checkpoint /path/to/Qwen2.5-Omni-7B
```
To enable FlashAttention-2, use the following command:
```bash
bash docker/docker_web_demo.sh --checkpoint /path/to/Qwen2.5-Omni-7B --flash-attn2
```

## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)



```BibTeX

@article{Qwen2.5-Omni,
  title={Qwen2.5-Omni Technical Report},
  author={Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, Junyang Lin},
  journal={arXiv preprint arXiv:2503.20215},
  year={2025}
}
```

<br>
