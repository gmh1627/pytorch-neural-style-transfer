# utils.py

import torch
from PIL import Image
import torchvision.transforms as transforms

def load_image(image_path, max_size=400, shape=None):
    image = Image.open(image_path).convert('RGB')
    
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
    
    if isinstance(size, tuple):
        in_transform = transforms.Compose([
            transforms.Resize(size),  # 直接使用元组
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))])
    else:
        in_transform = transforms.Compose([
            transforms.Resize((size, size)),  # 使用整数
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))])
    
    image = in_transform(image).unsqueeze(0)
    
    return image

def save_image(tensor, path):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(path)