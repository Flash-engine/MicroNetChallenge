# Flash MicroNet

Team name:Flash

Team members:  Xin Liu E-mail:750740751@qq.com
## Introduction

This repository contains our  solution of the [MicroNet Challenge](https://micronet-challenge.github.io/index.html) hosted at NeurIPS 2019.We compete in the Cifar100 track and we use the WideResNet28x10 as the baseline model .

Our micronet solution can be  broken down into two stages

1. **Train  from sratch**
3. **Quantization** 

With the above stages,our final micronet achieves <font color=red>80.25%</font> top-1 accuracy on Cifar100 dataset.The  final evaluation score is <font color=red>0.76348</font>.

## Requirements

+ Pytorch >=1.1.0
+ python 3.6 or above
+ tensorboardX==1.8

## Install

`git clone https://github.com/Flash-engine/MicroNetChallenge.git`

`cd MicroNetChallenge`

`git submodule update --remote`

Download the cifar100 dataset from [cifar100](https://www.cs.toronto.edu/~kriz/cifar.html) , extract the tar file  and place the extracted folder as you want.

## Detailed descriptions

In the following sections ,we will explain our solution by stages in details.

### 1.Train from scratch

We use the WideResnet28x10 model implementation provided by [meliketoy](https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py).The original WideResNet28x10 achieves <font color=red>80.75%</font> top-1 accuracy on Cifar-100 dataset.To further boost the network performance ,we incorporate the following strategies.

+ [**stochastic weight averaging**](https://arxiv.org/abs/1803.05407),[**reference implementation**](https://github.com/timgaripov/swa)
+ [**Cutout**](https://arxiv.org/abs/1708.04552),[**reference implementation**](https://github.com/uoguelph-mlrg/Cutout)
+ [**Mixup**](https://arxiv.org/abs/1710.09412),[**reference implementation**](https://github.com/facebookresearch/mixup-cifar10)

We refer to the implementations of  these three methods and organise them in **swa** submodule in this project folder.The  hyper-parameters of the strategies above used in our training are listed below 

+ **Stochastic weight averaging**

  This method is used in training to find a better optima and boost model generalization performance.

  Its hyper-parameters *swa_start* and *swa_lr* are set 161 and 0.05 separately.

+ **Cutout**

  **Cutout** randomly cuts a patch out of a sample image to augment the dataset.Its hyper-parameter  *n_holes* and  *length* are set to 1 and 16 separately in our experiment.

+ **Mixup**

  **Mixup** adopts a convex combination of taining samples and their labels to improve  generalization. Its hyper-parameter *alpha* is set to 1.0 in our experiment.

Our overall  training settings are  in the table below

| batchsize | optimizer | epochs | lr_init | weight decay | Momentum | Swa_lr | Swa_start |
| --------- | --------- | ------ | ------- | ------------ | -------- | ------ | --------- |
| 128       | SGD       | 300    | 0.1     | 5e-4         | 0.9      | 0.05   | 161       |

---

With the above strategies and settings ,our WideResnet28x10 achieves <font color=red>**82.68%**</font> top-1 accuracy  on Cifar-100 dataset.

To reproduce the reported accuracyfollow the steps below:

1. `cd swa`

2. In *run_train.sh*, specify *project_dir* , *dataset_dir* and *log_dir*

3. `./run_train.sh`



To evaluate the trained model

1. In *run_test.sh*, specify *project_dir* ,*dataset_dir* and *test_model* 

2. `./run_test.sh`

The already trained model is available in [swa_google_drive](https://drive.google.com/open?id=1krfv0vLvYWg4tylPqzL7dTWbcXmoBUNq)



### 2.Quantization

We use [DoReFa](https://arxiv.org/abs/1606.06160) to  quantize the trained model from the first stage and refer to  the implementation from [QuantizeCNNModel](https://github.com/nowgood/QuantizeCNNModel).<br> The quantization bit number *QUANTIZE_BIT* is set to be 4 in *${project_dir}/quantize/quantize_method.py* This means that all network layers except BN  layers are quantized into 4 bits.For each layer, inputs, weights, biases and activations are all quantized into 4 bits. 

You can find this setting in *${project_dir}/net/net_quantize_activation.py* line74~79. And our final quantized model still achieves <font color=red>80.25%</font>  top-1 accuracy on cifar100 dataset.

The quantization training settings are shown in the table below

| epochs | batch-size | mode | lr   | lr-step | momentum | Weight-decay | Seed |
| ------ | ---------- | ---- | ---- | ------- | -------- | ------------ | ---- |
| 100    | 256        | 3    | 0.01 | 10      | 0.9      | 1e-4         | 1    |

------

To reproduce our results, follow the steps below

1. `cd`  *QuantizeCNNModel*

2. In  *run_train.sh*,specify *root_dir*,*project_dir*,*dataset_dir* and *save-dir*

3. `./run_train.sh`



To evaluate the model

1. In *run_test.sh*,specify *evaluate* ,*load_trained_model* 

*evaluate* is the quantized model path

*load_trained_model* is the trained model path from the first stage

2. `./run_test.sh`

The already quantized model is availabel in [QuantizeCNNModel_google_drive](https://drive.google.com/open?id=1b2jsLFGNPoO3lfBEFpk7GIuzAHXElsLf)



## Scoring

In this section, we will descrip the calculation details of our final our model.

*count_wide_resnet.py* is the python file to count the *params* , *flops* and output the normalized final score.It refers to the official  [implementation](https://github.com/google-research/google-research/blob/master/micronet_challenge/counting.py). We omit the BN layers and the low-high precision conversions . Therefore, what we do is just collecting all the conv layers,fc layers and other operations like eletwise and pooling.

In *count_wide_resnet.py*,we set the *INPUT_BITS* ,*ACCUMULATOR_BITS* and *PARAMETER_BITS* to be 4 ,32 and 4 bits respectively.

Our final param score is **0.12510126(4.566196/36.5)** and flops score is **0.6383783672(6.69659/10.49)**.So our final score is **0.76348**

To get the score reported above ,follow the steps below

1. `cd model-statistics`

2. Modify *run_count.sh*,specify the *quantized _model* 

*quantized_model* is the quantized model path

3. run `python count_wide_resnet.py`



