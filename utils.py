# utils.py

from PIL import Image
import torch
import torchvision.transforms as transforms
import settings

# ImageNet mean and std
image_mean = torch.tensor([0.485, 0.456, 0.406])
image_std = torch.tensor([0.229, 0.224, 0.225])

def normalization(x):
    """
    对输入图片x进行归一化，返回归一化的值
    """
    return (x - image_mean[:, None, None]) / image_std[:, None, None]

def denormalization(x):
    """
    对输入图片x进行反归一化，返回反归一化的值
    """
    return x * image_std[:, None, None] + image_mean[:, None, None]

def load_image(image_path, width=settings.WIDTH, height=settings.HEIGHT):
    image = Image.open(image_path).convert('RGB')
    in_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()
    ])
    image = in_transform(image).unsqueeze(0)
    image = normalization(image)
    return image

def save_image(tensor, path):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = denormalization(image)
    image = unloader(image)
    image.save(path)