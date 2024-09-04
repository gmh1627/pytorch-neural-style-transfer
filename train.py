# train.py

import torch
import torch.optim as optim
from model import VGG
import settings
import utils
from tqdm import tqdm

# 加载内容和风格图片
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_image = utils.load_image(settings.CONTENT_IMAGE_PATH, shape=(settings.HEIGHT, settings.WIDTH)).to(device)
style_image = utils.load_image(settings.STYLE_IMAGE_PATH, shape=(settings.HEIGHT, settings.WIDTH)).to(device)

# 初始化噪声图片
generated_image = content_image.clone().requires_grad_(True)

# 加载模型
model = VGG().to(device).eval()

# 优化器
optimizer = optim.Adam([generated_image], lr=settings.LEARNING_RATE)

# 损失函数
mse_loss = torch.nn.MSELoss()

# 计算内容损失
def compute_content_loss(gen_features, orig_features):
    content_loss = mse_loss(gen_features, orig_features)
    return content_loss

# 计算风格损失
def compute_style_loss(gen_features, style_features):
    style_loss = 0
    for gen, style in zip(gen_features, style_features):
        _, c, h, w = gen.size()
        G = torch.mm(gen.view(c, h * w), gen.view(c, h * w).t())
        A = torch.mm(style.view(c, h * w), style.view(c, h * w).t())
        style_loss += mse_loss(G, A) / (c * h * w)
    return style_loss

# 训练过程
for epoch in range(settings.EPOCHS):
    with tqdm(total=settings.STEPS_PER_EPOCH, desc=f"Epoch {epoch+1}/{settings.EPOCHS}") as pbar:
        for step in range(settings.STEPS_PER_EPOCH):
            optimizer.zero_grad()
            
            gen_features = model(generated_image)
            orig_features = model(content_image)
            style_features = model(style_image)
            
            content_loss = compute_content_loss(gen_features[2], orig_features[2])
            style_loss = compute_style_loss(gen_features, style_features)
            
            total_loss = settings.CONTENT_LOSS_FACTOR * content_loss + settings.STYLE_LOSS_FACTOR * style_loss
            
            total_loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                pbar.set_postfix({
                    "Content Loss": f"{content_loss.item():.4f}",
                    "Style Loss": f"{style_loss.item():.4f}",
                    "Total Loss": f"{total_loss.item():.4f}"
                })
            
            pbar.update(1)
    
    # 保存生成的图片
    utils.save_image(generated_image, f"{settings.OUTPUT_DIR}/generated_{epoch+1}.png")