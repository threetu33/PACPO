# PACPO

## 项目简介

基于强化学习（PACPO）和大语言模型的推荐系统项目。支持使用Amazon Review数据集进行训练和评估。

## 目录结构概览

```
rrec/
├── train.py                    # PACPO强化学习训练主程序
├── test.py                     # 模型测试评估主程序
├── preprocess.py               # 数据预处理脚本
├── paths.py                    # 模型路径配置
├── models/                     # 模型定义
├── prompters/                  # Prompt生成器
├── trainers/                   # 训练器
└── requirements.txt            # Python依赖
```

## 安装依赖

`pip install -r requirements.txt`

若是中国用户，建议：
1. `pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1`
2. 注释掉 `torch==2.5.1+cu121`
3. `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`

## 训练流程

1. 数据处理：下载 amazon 数据集并且切分为训练集、验证集、测试集

> 数据集：Musical_Instruments, Video Games, Beauty
> 模型：Qwen2.5-3B-Instruct: [Download from Hugging Face](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

```bash
python preprocess.py \
      --category Musical_Instruments \
      --K 0 \
      --st_year 2022 \
      --st_month 10 \
      --ed_year 2023 \
      --ed_month 10 \
      --data_root_dir /path/to/your/dir
```

2. SFT 训练

3. RL 训练

(TODO，该部分取自RRec的README.md文件)

To train the model, simply run:

```bash
bash launch_train.sh
```

The script uses the following default configuration:
- Uses 4 GPUs (CUDA_VISIBLE_DEVICES=0,1,2,3)
- Runs 3 processes for distributed training
- Uses DeepSpeed for optimization
- Default dataset: Musical_Instruments
- Default model: Gemma-2-2b-it

Key parameters in the script:
- `NUM_PROCESSES`: Number of processes for distributed training
- `MAIN_PROCESS_PORT`: Port for the main process
- `DATASET_CAT`: Dataset category (e.g., "Musical_Instruments", "CDs_and_Vinyl")
- `DATASET_DIR`: Path to the processed dataset
- `MODEL`: Base model to use ("gemma" or "qwen")

Training hyperparameters:
- `train_batch_size`: 4
- `eval_batch_size`: 32
- `max_new_tokens`: 640
- `warmup_steps`: 32
- `num_train_epochs`: 3
- `group_size`: 4

You can modify these parameters in `launch_train.sh` according to your needs.


## 对比试验

1. **our_model_with_PACPO**

2. **our_model_with_GRPO**

3. **sasrec**

4. **gpu4rec**

5. **caser**

6. **Deepseek-R1**

7. **GLM-4.5**

8. **Kimi-K2**
