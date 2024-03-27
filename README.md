# Whisper

## 目录

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型与数据](#3-准备模型与数据)
* [4. 模型编译](#4-模型编译)
  * [4.1 TPU-MLIR编译BModel](#41-tpu-mlir编译bmodel)
* [5. 例程测试](#5-例程测试)
* [6. 精度测试](#6-精度测试)
  * [6.1 测试方法](#61-测试方法)
  * [6.2 测试结果](#62-测试结果)
* [7. 性能测试](#7-性能测试)
  * [7.1 bmrt_test](#71-bmrt_test)
  * [7.2 程序运行性能](#72-程序运行性能)
* [8. FAQ](#8-faq)

## 1. 简介
Whisper 是一个开源的深度学习语音识别模型，由 OpenAI 开发，它能够实现实时、多语言的语音识别，并支持跨多种环境和设备的灵活部署。本例程对[Whisper官方开源仓库](https://github.com/openai/whisper)中的算法进行移植，使之能在SOPHON BM1684X上进行推理。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)
* 支持FP16(BM1684X)模型编译和推理
* 支持基于SAIL推理的Python例程

## 3. 准备模型与数据
该模型目前只支持在1684X上运行，已提供编译好的bmodel，​同时，您需要准备用于测试的数据集。

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`。

```bash
# 安装unzip，若已安装请跳过
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```
./models
  └── BM1684X
      ├──bmwhisper_medium_1684x_f16.bmodel  # whisper-medium模型，模型参数量为769 M
      ├──bmwhisper_small_1684x_f16.bmodel  # whisper-small模型，模型参数量为244 M
      └──bmwhisper_base_1684x_f16.bmodel  # whisper-base模型，模型参数量为74 M
```

下载的数据包括：
```
./datasets
|── aishell_S0764                             # 从aishell数据集中抽取的用于测试的音频文件
|   └── *.wav
├── aishell_S0764.list                        # 从aishell数据集的文件列表
├── ground_truth.txt                          # 从aishell数据集的预测真实值
└── test                                      # 测试使用的音频文件
    └── demo.wav
```
## 4. 模型编译

可以直接下载我们已经导出的onnx模型，推荐在mlir部分提供的docker中完成转bmodel模型。
**注意**：
- 编译模型需要在x86主机完成。

### 4.1 TPU-MLIR环境搭建
模型编译前需要安装TPU-MLIR。安装好后需在TPU-MLIR环境中进入例程目录。先导出onnx，然后使用TPU-MLIR将onnx模型编译为BModel。编译的具体方法可参考《TPU-MLIR快速入门手册》的“3. 编译ONNX模型”(请从算能官网相应版本的SDK中获取)。

#### 4.1.1 安装docker
若已安装docker，请跳过本节。
```bash
# 安装docker
sudo apt-get install docker.io
# docker命令免root权限执行
# 创建docker用户组，若已有docker组会报错，没关系可忽略
sudo groupadd docker
# 将当前用户加入docker组
sudo usermod -aG docker $USER
# 切换当前会话到新group或重新登录重启X会话
newgrp docker​
```
> **提示**：需要logout系统然后重新登录，再使用docker就不需要sudo了。

#### 4.1.2 下载并解压TPU-MLIR
从sftp上获取TPU-MLIR压缩包
```bash
pip3 install dfss --upgrade
python3 -m dfss xxxxxxxxxxxxxxxxxxxxxxxx
```

#### 4.1.3 创建并进入docker
TPU-MLIR使用的docker是sophgo/tpuc_dev:latest, docker镜像和tpu-mlir有绑定关系，少数情况下有可能更新了tpu-mlir，需要新的镜像。
```bash
docker pull sophgo/tpuc_dev:latest
# 这里将本级目录映射到docker内的/workspace目录,用户需要根据实际情况将demo的目录映射到docker里面
# myname只是举个名字的例子, 请指定成自己想要的容器的名字
docker run --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:latest
# 此时已经进入docker，并在/workspace目录下
# 初始化软件环境
cd /workspace/tpu-mlir_vx.y.z-<hash>-<date>
source ./envsetup.sh
```
此镜像仅onnx模型导出和编译量化模型，程序编译和运行请在开发和运行环境中进行。更多TPU-MLIR的教程请参考算能官网的《TPU-MLIR快速入门手册》和《TPU-MLIR开发参考手册》。


### 4.2 获取onnx
从sftp上获取onnx模型
```bash
pip3 install dfss --upgrade
python3 -m dfss xxxxxxx
tar zxvf xxxxxxx
mv -r xxxxxxx ./models/onnx
```
下载的模型包括 whisper base/small/medium 的onnx模型：
```
./models
    └── onnx
      ├── decoder_loop_with_kvcache_base_5beam_448pad.onnx
      ├── decoder_loop_with_kvcache_medium_5beam_448pad.onnx
      ├── decoder_loop_with_kvcache_small_5beam_448pad.onnx
      ├── decoder_main_with_kvcache_base_5beam_448pad.onnx
      ├── decoder_main_with_kvcache_medium_5beam_448pad.onnx
      ├── decoder_main_with_kvcache_small_5beam_448pad.onnx
      ├── decoder_post_base_5beam_448pad.onnx
      ├── decoder_post_medium_5beam_448pad.onnx
      ├── decoder_post_small_5beam_448pad.onnx
      ├── encoder_base_5beam_448pad.onnx
      ├── encoder_medium_5beam_448pad.onnx
      ├── encoder_small_5beam_448pad.onnx
      ├── kvcache_rearrange_base_5beam_448pad.onnx
      ├── kvcache_rearrange_medium_5beam_448pad.onnx
      ├── kvcache_rearrange_small_5beam_448pad.onnx
      ├── logits_decoder_base_5beam_448pad.onnx
      ├── logits_decoder_medium_5beam_448pad.onnx
      └── logits_decoder_small_5beam_448pad.onnx
```

### 4.3 bmodel编译
目前TPU-MLIR支持1684x对Whisper进行F16量化，使用如下命令生成bmodel。
```bash
./scripts/gen_bmodel.sh --model base
```
其中，model可以指定base/small/medium，编译成功之后共有6个模型放置于`./models/1684X/'，需要将多个模型combine。

在装有驱动的docker/盒子/PCIE主机环境执行以下命令，以base为例
```bash
sudo chmod -R 777 ./models/1684X/
cd ./models/1684X/
tpu_model --combine all_quant_encoder_base_5beam_448pad_1684x_f16.bmodel all_quant_logits_decoder_base_5beam_448pad_1684x_f16.bmodel all_quant_decoder_main_with_kvcache_base_5beam_448pad_1684x_f16.bmodel all_quant_decoder_post_base_5beam_448pad_1684x_f16.bmodel all_quant_decoder_loop_with_kvcache_base_5beam_448pad_1684x_f16.bmodel all_quant_kvcache_rearrange_base_5beam_448pad_1684x_f16.bmodel -o bmwhisper_base_1684x_f16.bmodel
```

最终会生成模型`bmwhisper_base_1684x_f16.bmodel`


## 5. 例程测试

- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法
首先，参考[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测结果至result路径，注意修改数据集(datasets/aishell_S0764)和相关参数。
然后，使用`tools`目录下的`eval_aishell.py`脚本，将测试生成的txt文件与测试集标签txt文件进行对比，计算出语音识别的评价指标，命令如下：
```bash
# 请根据实际情况修改程序路径和txt文件路径
python3 tools/eval_aishell.py --char=1 --v=1 datasets/aishell_S0764/ground_truth.txt python/result  > online_wer
cat online_wer | grep "Overall"
```

### 6.2 测试结果
在aishell数据集上，精度测试结果如下：
|   测试平台    |    测试程序   |              测试模型                                 | WER    |
| ------------ | ------------ | ----------------------------------------------------- | ------ |
| BM1684X Pcie | run.py       | bmwhisper_base_1684x_f16.bmodel                       | 2.70%  |
| BM1684X Pcie | run.py       | bmwhisper_small_1684x_f16.bmodel                      | 2.70%  |
| BM1684X Pcie | run.py       | bmwhisper_medium_1684x_f16.bmodel                     | 2.70%  |
| BM1684X SoC  | run.py       | bmwhisper_base_1684x_f16.bmodel                       | 3.45%  |
| BM1684X SoC  | run.py       | bmwhisper_small_1684x_f16.bmodel                      | 3.45%  |
| BM1684X SoC  | run.py       | bmwhisper_medium_1684x_f16.bmodel                     | 3.45%  |

> **测试说明**：
1. 在使用的模型相同的情况下，wer在不同的测试平台上是相同的。
2. 由于SDK版本之间的差异，实测的wer与本表有1%以内的差值是正常的。

## 7. 性能测试
|    测试平台   |     测试程序      |           测试模型                  |tpu inference time(s) |cpu inference time(s)    |
| -----------  | ---------------- | -----------------------------------| --------------------- | ----------------------- |
| BM1684X Pcie | run.py           | bmwhisper_base_1684x_f16.bmodel    | 1.21                  | 3.96                    |
| BM1684X Pcie | run.py           | bmwhisper_small_1684x_f16.bmodel   | 0.89                  | 8.64                    |
| BM1684X Pcie | run.py           | bmwhisper_medium_1684x_f16.bmodel  | 0.89                  | 14.24                   |
| BM1684X SoC  | run.py           | bmwhisper_base_1684x_f16.bmodel    | 1.59                  | 3.78                    |
| BM1684X SoC  | run.py           | bmwhisper_small_1684x_f16.bmodel   | 1.22                  | 8.12                    |
| BM1684X SoC  | run.py           | bmwhisper_medium_1684x_f16.bmodel  | 1.22                  | 13.14                   |

> **测试说明**：
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. BM1684X SoC的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz。