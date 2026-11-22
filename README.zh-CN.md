# SAFL-Net

面向乳腺超声病灶分割与良恶性分类的联合训练和评估流程。

[English](README.md)

## 项目简介

项目使用配对超声图像、分割标注及二分类标签，提供基于显式列表的数据读取、
配对增强、点与框提示、联合损失、检查点选择和逐病例结果导出功能。

## 环境要求

- Python 3.9
- PyTorch 2.4.1、torchvision 0.19.1
- `requirements.txt` 中的依赖
- 按下文配置可导入的模型后端
- GPU 训练需要 CUDA；流程工具支持 CPU

## 安装

```bash
git clone https://github.com/zhl05/SAFL-Net.git
cd SAFL-Net
python -m venv .venv
```

Linux/macOS 使用 `source .venv/bin/activate`，Windows PowerShell 使用
`.venv\Scripts\Activate.ps1` 激活环境。根据平台按照
[PyTorch 安装说明](https://pytorch.org/get-started/previous-versions/#v241)
安装对应的 PyTorch/torchvision 版本，然后安装其余依赖：

```bash
python -m pip install -r requirements.txt
```

## 目录结构

```text
SAFL-Net/
├── safl/
│   ├── backend.py          # 模型接口
│   ├── data.py             # 数据读取、配对变换及提示
│   ├── engine.py           # 训练与评估流程
│   ├── losses.py           # 分割与分类损失
│   ├── metrics.py          # 评估指标
│   └── io.py               # 检查点与结果保存
├── examples/               # 数据列表格式示例
├── tests/                  # 合成数据验证
├── train.py                # 单折训练入口
├── test.py                 # 测试入口
├── evaluate.py             # test.py 的兼容入口
└── requirements.txt
```

## 数据准备

数据放在源码目录之外。图像与分割标注均为灰度 PNG，原始尺寸应一致，
标注中的非零像素视为前景。

```text
data_root/
└── BUSBRA/
    ├── Image/example_case.png
    └── Label/example_case.png
```

UTF-8 数据列表每行包含四个字段，病例名不包含 `.png` 扩展名：

```text
前景编号/数据集子目录/病例名/分类标签
1/BUSBRA/example_case/0
```

前景编号固定为 `1`；分类标签 `0` 表示良性、`1` 表示恶性，使用前请核对
标注映射。允许空行和以 `#` 开头的注释。程序拒绝重复病例及训练/验证列表的
病例交叉。患者级划分需自行保证，程序不会推断患者身份。
`examples/` 中的病例名仅用于说明格式。

训练前检查图像、标注及提示生成：

```bash
python -m safl.data --data-root /path/to/data_root --manifest /path/to/fold1_val.txt
```

## 模型配置

运行环境应提供可导入的 `safl_model` 模块，也可通过 `SAFL_BACKEND` 环境变量
指定其他模块名。`safl/backend.py` 使用以下接口：

- `build_model(image_size=256, low_image_size=128)`：返回 PyTorch 模型。
- `build_optimizer(model)`：返回 `(optimizer, scheduler)`。`scheduler` 可为
  `None`；否则在每次成功更新参数后调用其 `step()`。

模型调用形式为 `model(image, points, boxes, use_auto_prompt=False)`，返回
`(segmentation, class_logits, auxiliary_1, auxiliary_2)`。
`segmentation` 包含尺寸为 `B×1×256×256` 的 `masks` logits，以及尺寸为
`B×1×128×128` 的 `low_res_logits`；分类 logits 尺寸为 `B×2`。

## 训练

每次运行训练一折，分别指定该折训练与验证列表：

```bash
python train.py --data-root /path/to/data_root --train-manifest /path/to/fold1_train.txt --val-manifest /path/to/fold1_val.txt --fold 1 --output outputs/fold1
```

默认训练 100 个 epoch、batch size 为 8、随机种子为 1234。
使用 `--device cpu` 可切换至 CPU，完整参数见 `python train.py --help`。
输出目录应为新目录或空目录。五折分别使用各自列表及独立输出目录运行。

训练保存 `foldN_best_dice.pth`、`foldN_best_auc.pth`、对应验证预测 CSV 及
`history.json`。主指标相同时以另一指标打破平局；AUC 选择需要验证集同时
包含两类标签。检查点包含模型权重和选择信息，不作为中断恢复快照。

## 测试

对独立准备的评估列表运行匹配的检查点：

```bash
python test.py --data-root /path/to/data_root --manifest /path/to/test.txt --checkpoint /path/to/fold1_best_auc.pth --output outputs/test_fold1 --save-masks
```

`evaluate.py` 接受相同参数，完整选项见 `python test.py --help`。
权重按严格参数匹配方式加载。输出目录应为新目录或空目录。
默认生成 `metrics.json` 和 `predictions.csv`；添加 `--save-masks` 后还保存
二值 PNG 掩膜并计算 HD95/ASSD。

## 指标与评估约定

- 图像及完整标注缩放至 `256×256`，损失标注为 `128×128`。
- 训练与评估均使用由标注掩膜生成的点和框。评估需要参考标注，使用固定点及紧致框。
- CUDA 训练使用 AMP，评估使用 FP32。
- 分割阈值为 sigmoid 概率 `> 0.5`；分类阈值为类别 1 的 softmax 概率
  `>= 0.5`。AUC 使用连续概率，单类评估集的 AUC 输出为 `null`。
- Dice/IoU 按病例平均，平滑项为 `1e-5`。分类指标包含 AUC、Accuracy、F1、
  Precision、Sensitivity 和 Specificity。
- HD95/ASSD 基于双向边界距离，单位为缩放后图像像素。预测与标注均为空时
  距离为零；仅一方为空的病例单独计数，不纳入距离均值。
- 用于检查点选择的数据应报告为验证集；评估列表应独立于模型选择确定。

## 验证

```bash
python -m unittest discover -s tests -v
```

验证使用合成数据，检查流程工具和命令行行为，不衡量分割或分类性能。

## 致谢与使用条款

上游项目致谢见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，使用条款见
[RIGHTS.md](RIGHTS.md)。数据集的获取及使用需遵循相应数据条款。
