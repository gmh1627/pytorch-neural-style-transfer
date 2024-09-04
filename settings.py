# settings.py

# 内容特征层及loss加权系数
CONTENT_LAYERS = {'conv_4': 1.0}
# 风格特征层及loss加权系数
STYLE_LAYERS = {'conv_1': 0.2, 'conv_2': 0.2, 'conv_3': 0.2, 'conv_4': 0.2, 'conv_5': 0.2}
# 内容图片路径
CONTENT_IMAGE_PATH = './images/content.jpg'
# 风格图片路径
STYLE_IMAGE_PATH = './images/style.jpg'
# 生成图片的保存目录
OUTPUT_DIR = './output'

# 内容loss总加权系数
CONTENT_LOSS_FACTOR = 1
# 风格loss总加权系数
STYLE_LOSS_FACTOR = 100

# 图片宽度
WIDTH = 450
# 图片高度
HEIGHT = 300

# 训练epoch数
EPOCHS = 100
# 每个epoch训练多少次
STEPS_PER_EPOCH = 100
# 学习率
LEARNING_RATE = 0.03