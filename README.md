# MT5_paddle
Use PaddlePaddle to reproduce the paper：mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer
English | [简体中文](./README_cn.md)


[mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer](https://arxiv.org/abs/2010.11934)



**Abstract：**
The recent “Text-to-Text Transfer Transformer” (T5) leveraged a unified text-to-text format and scale to attain state-of-the-art results on a wide variety of English-language
NLP tasks. In this paper, we introduce mT5, a multilingual variant of T5 that was pre-trained on a new Common Crawl-based dataset covering 101 languages. We detail the design and modified training of mT5 and demonstrate its state-of-the-art performance on many multilingual benchmarks. We also describe a simple technique to prevent “accidental translation” in the zero-shot setting, where a generative model chooses to (partially) translate its prediction into the wrong language. All of the code and model checkpoints used in this work are publicly available.

This project is an open source implementation of MT5 on Paddle 2.x.


## Environment Installation

| label  | value     |
|--------|------------------|
| python | >=3.6     |
| GPU    | V100       |
| Frame    | PaddlePaddle2\.1.2 |
| Cuda   | 10.1         |
| Cudnn  | 7.6 |

Cloud platform used in this recurrence：https://aistudio.baidu.com/




```bash
# Clone the repository
git clone https://github.com/27182812/ChineseBERT_paddle
# Enter the root directory
cd ChineseBERT_paddle
# Install the necessary python libraries locally
pip install -r requirements.txt

```

## Quick Start

### （一）Model Accuracy Alignment
run `python compare.py`，Comparing the accuracy between huggingface and paddle, we can find that the average error of accuracy is on the order of 10^-7, and the maximum error is on the order of 10^-6.
```python
python compare.py
# ChineseBERT-large-pytorch vs paddle ChineseBERT-large-paddle
mean difference: 3.5541183e-07
max difference: 7.6293945e-06
# ChineseBERT-base-pytorch vs paddle ChineseBERT-base-paddle
mean difference: 2.1572937e-06
max difference: 4.3660402e-05

```

<img src="imgs/4.png" width="80%" />
<img src="imgs/5.png" width="80%" />

#### Pre-trained model weights-base

链接：https://pan.baidu.com/s/1eclrM8ahm6Fiz-gkGHYpZg 
提取码：8gdx

#### Pre-trained model weights-large

链接：https://pan.baidu.com/s/1sVhWS96tIDZOx-2fk9mgpw 
提取码：i07w



#### Model weight, dictionary and tokenizer_config path configuration instructions

##### Pre-training weights

将[modeling.py](pdchinesebert/modeling.py)中第81行ChineseBERT-large对应的路径改为权重实际的路径

##### Dictionary path

将[tokenizer.py](pdchinesebert/tokenizer.py)中第10行ChineseBERT-large对应的字典路径改为vocab.txt实际所在的路径

##### Tokenizer_config path

将[tokenizer.py](pdchinesebert/tokenizer.py)中第14行ChineseBERT-large对应的路径改为tokenizer_config.json实际所在路径




### （二）Downstream task fine-tuning

#### 1、ChnSentiCorp
Take the ChnSentiCorp dataset as an example.

#### （1）Model fine-tuning：
```shell
# run train
python train_chn.py \
--data_path './data/ChnSentiCorp' \
--device 'gpu' \
--epochs 10 \
--max_seq_length 512 \
--batch_size 8 \
--learning_rate 2e-5 \
--weight_decay 0.0001 \
--warmup_proportion 0.1 \
--seed 2333 \
--save_dir 'outputs/chn' | tee outputs/train_chn.log
```
**Model Link**

link：https://pan.baidu.com/s/1DKfcUuPxc7Kymk__UXHvMw 
password：85rl

#### (2) Evaluate

The acc on the dev and test datasets are respectively 95.8 and 96.08, which meet the accuracy requirements of the paper. The results are as follows:

<p>
 <img src="imgs/chn_test.png" width="66%" /> 
</p>

#### 2、XNLI

#### （1）Train

```bash
python train_xnli.py \
--data_path './data/XNLI' \
--device 'gpu' \
--epochs 5 \
--max_seq_len 256 \
--batch_size 16 \
--learning_rate 1.3e-5 \
--weight_decay 0.001 \
--warmup_proportion 0.1 \
--seed 2333 \
--save_dir outputs/xnli | tee outputs/train_xnli.log
```

#### （2）Evaluate

The best result of the test dataset acc is 81.657, which meets the accuracy requirements of the paper. The results are as follows:

<img src="imgs/xnli_test.png" width="66%" />

**Model Link**

link：https://pan.baidu.com/s/1lZ2T31FlZecKSOEHwExbrQ 
password：oskm

#### 3、cmrc2018

#### (1) Train

```shell
# Start train
python train_cmrc2018.py \
    --model_type chinesebert \
    --data_dir "data/cmrc2018" \
    --model_name_or_path ChineseBERT-large \
    --max_seq_length 512 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --eval_batch_size 16 \
    --learning_rate 4e-5 \
    --max_grad_norm 1.0 \
    --num_train_epochs 3 \
    --logging_steps 2 \
    --save_steps 20 \
    --warmup_radio 0.1 \
    --weight_decay 0.01 \
    --output_dir outputs/cmrc2018 \
    --seed 1111 \
    --num_workers 0 \
    --use_amp
```

During the training process, the model will be evaluated on the dev dataset, and the best results are as follows:

```python

{
    AVERAGE = 82.791
    F1 = 91.055
    EM = 74.526
    TOTAL = 3219
    SKIP = 0
}

```

<img src="imgs/cmrcdev.png" width="80%" />


#### （2）Run eval.py to generate the test data set to predict the answer

```bash
python eval.py --model_name_or_path outputs/step-340 --n_best_size 35 --max_answer_length 65
```

Among them, model_name_or_path is the model path

#### （3）Submit to CLUE

The test dataset EM is 78.55, which meets the accuracy requirements of the paper. The results are as follows:

<p align="center">
 <img src="imgs/cmrctest.png" width="100%"/>
</p>

**Model Link**

link：https://pan.baidu.com/s/11XSY3PPB_iWNBVme6JmqAQ 
password：17yw



### Train Log

Training logs  can be find [HERE](logs)




# Reference

```bibtex
@article{sun2021chinesebert,
  title={ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information},
  author={Sun, Zijun and Li, Xiaoya and Sun, Xiaofei and Meng, Yuxian and Ao, Xiang and He, Qing and Wu, Fei and Li, Jiwei},
  journal={arXiv preprint arXiv:2106.16038},
  year={2021}
}

```
