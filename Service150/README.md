---
license: Apache License 2.0
audio:
  text-to-speech:
    language:
      - zh
    sampling_rate:
      - "48000"
    style:
      - custom-service
    emotion:
      - neutral
tags:
  - TTS
---
# 数据堂150人中文客服平均音色合成库

## 数据集描述
用于150人中文客服平均音色合成库  中文客服平均音色合成库模型”模型的测试任务

### 数据集简介

150人中文客服平均音色合成库，由中文母语发音人录制，客服场景的录音文本，语料音素覆盖均衡，专业语音学家参与标注，精准匹配语音合成的研发需求。

### 数据集支持的任务
中文客服平均音色合成库模型”模型的测试任务

## 数据集的格式和结构

### 数据格式
48kHz，16bit，wav，单声道

### 人员
150人，20~30岁

### 录音内容
客服场景的录音文本，音节音素音调都进行了平衡覆盖

## 数据集生成的相关信息

### 原始数据
无

### 数据集标注
句准确率不低于95%


#### 标注特点
音字标注、四级韵律标注

#### 标注者
无


## 数据集版权信息
版权归数所堂所有，商用数据。


## 其他相关信息
详见https://www.datatang.com/dataset/1100?source=modelscope

### Clone with HTTP
```bash
git clone https://www.modelscope.cn/datasets/DatatangBeijing/150People-ChineseMandarinAverageToneSpeechSynthesisCorpus-CustomerService.git
```