# settings.py

# 内容特征层及loss加权系数
CONTENT_LAYERS = {'21': 0.5, '30': 0.5}  # 对应于 'block4_conv2' 和 'block5_conv2'
# 风格特征层及loss加权系数
STYLE_LAYERS = {'0': 0.2, '5': 0.2, '10': 0.2, '19': 0.2, '28': 0.2}  # 对应于 'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'
# 内容图片路径
CONTENT_IMAGE_PATH = './images/input.jpg'
# 风格图片路径
STYLE_IMAGE_PATH = './images/style2.jpg'
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
EPOCHS = 20
# 每个epoch训练多少次
STEPS_PER_EPOCH = 100
# 学习率
LEARNING_RATE = 0.01