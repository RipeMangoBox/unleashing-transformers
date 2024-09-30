import torch
# # 实例化模型
# model = SimpleCNN()

# 加载 checkpoint
checkpoint = torch.load('checkpoints/vqgan_ffhq/vqgan_1400000.th')


# generator_ckpt = {(k, v) for k, v in checkpoint.items() if 'generator' in k}

# 初始化空的字典来存储各个组件的参数
encoder_state_dict = {}
quantizer_state_dict = {}
generator_state_dict = {}

# 遍历状态字典，根据键名提取参数
for key, value in checkpoint.items():
    if key.startswith('ae.encoder'):
        encoder_state_dict[key] = value
    elif key.startswith('ae.quantize'):
        quantizer_state_dict[key] = value
    elif key.startswith('ae.generator'):
        generator_state_dict[key] = value

# 保存 encoder 的参数
torch.save({'model_state_dict': encoder_state_dict}, 'encoder_checkpoint.pt')

# 保存 quantizer 的参数
torch.save({'model_state_dict': quantizer_state_dict}, 'quantizer_checkpoint.pt')

# 保存 generator 的参数
torch.save({'model_state_dict': generator_state_dict}, 'generator_checkpoint.pt')
# # 将状态字典加载到模型中
# model.load_state_dict(checkpoint['model_state_dict'])

# # 设置模型为评估模式
# model.eval()