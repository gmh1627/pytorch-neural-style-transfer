# pytorch-neural-style-transfer
运用pytorch和深度学习实现图像的神经风格迁移
## 1.各代码文件详解
### 1.1 `train.py`
`train.py` 文件负责训练神经风格迁移模型。
- **加载内容和风格图片**：使用 `utils.load_image` 函数加载并预处理内容和风格图片。
- **初始化生成图像**：将内容图像加上随机噪声作为初始生成图像。
- **加载模型**：实例化并加载神经风格迁移模型。
- **设置优化器和损失函数**：使用 Adam 优化器和均方误差损失函数。
- **定义内容损失和风格损失的计算函数**：包括 `_compute_content_loss`, `compute_content_loss`, `gram_matrix`, `_compute_style_loss`, `compute_style_loss`, 和 `total_loss`。
- **计算目标内容图片和风格图片的特征**：通过模型提取内容和风格特征。
- **创建保存生成图片的文件夹**：检查并创建输出目录。
- **训练过程**：使用 `tqdm` 显示训练进度条，进行多轮训练，每轮训练后保存生成的图片。

### 1.2 `model.py`
`model.py` 文件定义了神经风格迁移模型。
- **定义获取 VGG19 模型的函数**：`get_vgg19_model` 函数从预训练的 VGG19 模型中提取指定层。
- **定义神经风格迁移模型类**：`NeuralStyleTransferModel` 类继承自 `nn.Module`，包含模型的初始化和前向传播方法。

### 1.3 `utils.py`
`utils.py` 文件包含图像处理的辅助函数。
- **定义图像归一化和反归一化函数**：`normalization` 和 `denormalization` 函数对图像进行归一化和反归一化处理。
- **定义加载和保存图像的函数**：`load_image` 函数加载并预处理图像，`save_image` 函数保存生成的图像。

### 1.4 `settings.py`
`settings.py` 文件包含训练过程中的各种配置参数。
- **定义各种配置参数**：包括内容图像路径、风格图像路径、输出目录、图像宽度和高度、学习率、训练轮数、每轮训练步数、内容损失和风格损失的权重因子、内容层和风格层的配置。

### 2.环境要求
- **操作系统**：Windows, macOS, 或 Linux
- **Python 版本**：Python 3.6 及以上
- **依赖库**：
  - `torch`：用于深度学习模型的构建和训练
  - `torchvision`：用于图像处理和预训练模型
  - `PIL` (或 `Pillow`)：用于图像加载和保存
  - `tqdm`：用于显示训练进度条
  
### 3.效果展示
详见[这篇文章](https://blog.csdn.net/weixin_73004416/article/details/141905688)