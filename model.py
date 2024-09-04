# model.py

import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG19_Weights
import settings
import typing

def get_vgg19_model(layers):
    """
    创建并初始化VGG19模型，并在指定层添加Identity层以提取输出
    :param layers: 需要提取的层的名称列表
    :return: 提取指定层输出的VGG19模型
    """
    # 加载预训练的VGG19模型
    vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
    model = nn.Sequential()
    # 遍历VGG19模型的每一层
    for i, layer in enumerate(vgg):
        # 将每一层添加到新的Sequential模型中
        model.add_module(str(i), layer)
        # 如果当前层在需要提取的层列表中，添加一个Identity层
        if str(i) in layers:
            model.add_module(f"output_{i}", nn.Identity())
    return model

class NeuralStyleTransferModel(nn.Module):
    def __init__(self, content_layers: typing.Dict[str, float] = settings.CONTENT_LAYERS,
                 style_layers: typing.Dict[str, float] = settings.STYLE_LAYERS):
        """
        初始化神经风格迁移模型
        :param content_layers: 内容特征层及其loss加权系数
        :param style_layers: 风格特征层及其loss加权系数
        """
        super(NeuralStyleTransferModel, self).__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers
        # 合并内容层和风格层的名称
        layers = list(self.content_layers.keys()) + list(self.style_layers.keys())
        # 创建层名称到索引的映射
        self.outputs_index_map = {layer: i for i, layer in enumerate(layers)}
        # 获取VGG19模型
        self.vgg = get_vgg19_model(layers)
        # 设置模型为评估模式
        self.vgg.eval()
        # 冻结VGG19模型的参数
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        前向传播，提取指定层的输出
        :param x: 输入图像
        :return: 包含内容特征和风格特征的字典
        """
        outputs = []
        # 遍历VGG19模型的每一层
        for i, (name, layer) in enumerate(self.vgg.named_children()):
            # 将输入图像通过当前层
            x = layer(x)
            # 如果当前层是需要提取输出的层，保存输出
            if f"output_{i}" in self.vgg._modules:
                outputs.append(x)
        # 提取内容特征
        content_outputs = [(outputs[self.outputs_index_map[layer]], factor) for layer, factor in self.content_layers.items()]
        # 提取风格特征
        style_outputs = [(outputs[self.outputs_index_map[layer]], factor) for layer, factor in self.style_layers.items()]
        # 返回包含内容特征和风格特征的字典
        return {'content': content_outputs, 'style': style_outputs}