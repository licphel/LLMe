# LLMe - 个人语言模型训练器

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

LLMe 是一个从零实现的轻量级语言模型训练框架，支持多格式数据加载、对话训练和模型部署。让你在个人电脑上就能训练自己的对话模型！

## 特性

- **轻量级设计** - 1M-8M 参数，个人电脑轻松运行
- **多格式支持** - 自动识别 TXT、Alpaca、ShareGPT、MOSS 格式
- **HuggingFace 集成** - 一键下载和加载 HF 数据集
- **对话式训练** - 支持多轮对话，自动添加特殊标记
- **早停机制** - 自动监控 loss，防止过拟合
- **断点续训** - 随时中断，随时继续
- **实时监控** - 训练速度、loss、学习率一目了然

## 快速开始

### 1. 安装依赖

使用conda或pip按照requirements.txt安装依赖。推荐使用python3.9

### 2. 准备数据

将你的数据放在 `data/` 目录下，支持格式：
- `.txt` - 纯文本
- `.json`/`.jsonl` - Alpaca、ShareGPT、MOSS 格式

或者从 HuggingFace 下载：
```bash
/fetch hf <Name>
```

### 3. 启动聊天界面

```bash
python chat.py
```

### 4. 可用命令

| 命令 | 说明 | 示例 |
|------|------|------|
| `/load <path>` | 加载数据到缓存区 | `/load data/` |
| `/clear` | 清空缓存区 | `/clear` |
| `/train <name>` | 训练新模型（使用缓存区数据） | `/train <Name>` |
| `/resume <name>` | 继续训练（使用缓存区数据） | `/resume <Name> <Epochs>` |
| `/switch <name>` | 切换模型 | `/switch <Name>` |
| `/fetch hf <name>` | 下载 HuggingFace 数据集（到fetch文件夹） | `/fetch hf <Name>` |
| `/quit` | 退出 | `/quit` |

## 配置参数

编辑 `config/settings.json`：

```json
{
    "max_sequence_length": 256,    // 序列长度
    "stride": 128,                 // 滑动步长
    "dimensions": 128,             // 嵌入维度
    "layers": 4,                   // Transformer 层数
    "heads": 4,                    // 注意力头数
    "learning_rate": 0.0002,       // 学习率
    "epochs": 5,                   // 训练轮数
    "batch_size": 8,               // 批次大小
    "max_length": 512,             // 生成长度
    "temperature": 0.7             // 生成温度
}
```

## 贡献

欢迎提交 Issue 和 PR！

## 许可证

[MIT License](LICENSE)