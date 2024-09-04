import os
import torch
import torch.optim as optim
from model import NeuralStyleTransferModel
import settings
import utils
from tqdm import tqdm

# 加载内容和风格图片
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_image = utils.load_image(settings.CONTENT_IMAGE_PATH, width=settings.WIDTH, height=settings.HEIGHT).to(device)
style_image = utils.load_image(settings.STYLE_IMAGE_PATH, width=settings.WIDTH, height=settings.HEIGHT).to(device)

# 初始化生成图像为内容图像加上随机噪声
noise = torch.randn_like(content_image) * 0.1
generated_image = torch.nn.Parameter((content_image + noise) / 2)

# 加载模型
model = NeuralStyleTransferModel().to(device)

# 优化器
optimizer = optim.Adam([generated_image], lr=settings.LEARNING_RATE)

# 损失函数
mse_loss = torch.nn.MSELoss()

# 计算内容损失
def _compute_content_loss(noise_features, target_features):
    content_loss = mse_loss(noise_features, target_features)
    x = 2. * settings.WIDTH * settings.HEIGHT * 3
    return content_loss / x

def compute_content_loss(noise_content_features, target_content_features):
    content_losses = []
    for (noise_feature, factor), (target_feature, _) in zip(noise_content_features, target_content_features):
        layer_content_loss = _compute_content_loss(noise_feature, target_feature)
        content_losses.append(layer_content_loss * factor)
    return sum(content_losses)

# 计算格拉姆矩阵
def gram_matrix(feature):
    _, c, h, w = feature.size()
    x = feature.view(c, h * w)
    return torch.mm(x, x.t())

# 计算风格损失
def _compute_style_loss(noise_feature, target_feature):
    noise_gram_matrix = gram_matrix(noise_feature)
    style_gram_matrix = gram_matrix(target_feature)
    style_loss = mse_loss(noise_gram_matrix, style_gram_matrix)
    x = 4. * (settings.WIDTH ** 2) * (settings.HEIGHT ** 2) * (3 ** 2)
    return style_loss / x

def compute_style_loss(noise_style_features, target_style_features):
    style_losses = []
    for (noise_feature, factor), (target_feature, _) in zip(noise_style_features, target_style_features):
        layer_style_loss = _compute_style_loss(noise_feature, target_feature)
        style_losses.append(layer_style_loss * factor)
    return sum(style_losses)

def total_loss(noise_features, target_content_features, target_style_features):
    content_loss = compute_content_loss(noise_features['content'], target_content_features)
    style_loss = compute_style_loss(noise_features['style'], target_style_features)
    return content_loss * settings.CONTENT_LOSS_FACTOR + style_loss * settings.STYLE_LOSS_FACTOR

# 计算目标内容图片和风格图片的特征
target_content_features = model(content_image)['content']
target_style_features = model(style_image)['style']

# 创建保存生成图片的文件夹
if not os.path.exists(settings.OUTPUT_DIR):
    os.mkdir(settings.OUTPUT_DIR)

# 训练过程
for epoch in range(settings.EPOCHS):
    with tqdm(total=settings.STEPS_PER_EPOCH, desc=f"Epoch {epoch+1}/{settings.EPOCHS}") as pbar:
        for step in range(settings.STEPS_PER_EPOCH):
            optimizer.zero_grad()
            
            noise_outputs = model(generated_image)
            loss = total_loss(noise_outputs, target_content_features, target_style_features)
            
            loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}"
                })
            
            pbar.update(1)
    
    # 保存生成的图片
    utils.save_image(generated_image, f"{settings.OUTPUT_DIR}/generated_{epoch+1}.png")