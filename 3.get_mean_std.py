from hparams import get_sampler_hparams
import os
from utils.sampler_utils import generate_latent_ids, get_latent_loaders, retrieve_autoencoder_components_state_dicts,\
    get_samples, get_sampler
from models import VQAutoEncoder, Generator
from utils.data_utils import get_data_loaders, cycle
from utils.log_utils import log, log_stats, set_up_visdom, config_log, start_training_log, \
    save_stats, load_stats, save_model, load_model, save_images
    
import torch
import numpy as np
        
def load(H):
    latents_fp_suffix = '_flipped' if H.horizontal_flip else ''
    latents_filepath = f'latents/{H.dataset}_{H.latent_shape[-1]}_train_latents{latents_fp_suffix}'

    train_with_validation_dataset = False
    if H.steps_per_eval:
        train_with_validation_dataset = True

    if not os.path.exists(latents_filepath):
        ae_state_dict = retrieve_autoencoder_components_state_dicts(
            H, ['encoder', 'quantize', 'generator']
        )
        ae = VQAutoEncoder(H)
        ae.load_state_dict(ae_state_dict, strict=False)
        # val_loader will be assigned to None if not training with validation dataest
        train_loader, val_loader = get_data_loaders(
            H.dataset,
            H.img_size,
            H.batch_size,
            drop_last=False,
            shuffle=False,
            get_flipped=H.horizontal_flip,
            get_val_dataloader=train_with_validation_dataset
        )

        print("Transferring autoencoder to GPU to generate latents...")
        ae = ae.cuda(0)  # put ae on GPU for generating
        generate_latent_ids(H, ae, train_loader, val_loader)
        print("Deleting autoencoder to conserve GPU memory...")
        ae = ae.cpu()
        ae = None

    train_latent_loader, val_latent_loader = get_latent_loaders(H, get_validation_loader=train_with_validation_dataset, shuffle=False) # [473, 256]

    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['quantize', 'generator'],
        remove_component_from_key=True
    )

    embedding_weight = quanitzer_and_generator_state_dict.pop(
        'embedding.weight')
    # if H.deepspeed:
    #     embedding_weight = embedding_weight.half()
    embedding_weight = embedding_weight.cuda(0)
    generator = Generator(H)

    generator.load_state_dict(quanitzer_and_generator_state_dict, strict=False)
    generator = generator.cuda(0)
    sampler = get_sampler(H, embedding_weight).cuda(0)
    # sampler = load_model(sampler, H.sampler, H.load_step, H.load_dir).cuda(0) # inference
    
    train_iterator = cycle(train_latent_loader)
    # val_iterator = cycle(val_latent_loader)

    print(f"Sampler params total: {sum(p.numel() for p in sampler.parameters())}")
    
    # for i in range(0, len(train_dataset)):
    #     if torch.all(train_dataset[i] == train_dataset[0]):
    #         print(f'equal data: {i}')
    # print()
    
    start_step = 0
    train_dataset = []
    for step in range(start_step, 473):
        x = next(train_iterator)
        x = x.cuda(0)
        print(f"step: {step}")
        train_dataset.append(x)
    
    
    float_latents = []
    for i in range(0, len(train_dataset)):
        idx = train_dataset[i]
        N, C = idx.shape
        latents_one_hot = sampler.latent_ids_to_onehot(idx)
        zs = sampler.embed(latents_one_hot).view(N, C, -1) # [b, embed_dim]
        float_latents.append(zs)
        
    # 将 train_dataset 转换为一个张量
    # 假设 train_dataset 是一个列表，且每个元素的形状为 (256, H, W)
    train_tensor = torch.cat(float_latents, dim=0)  # 合并到一个大张量

    # 计算 mean 和 std
    mean = train_tensor.mean(dim=(0))  # 沿着 H 和 W 轴计算均值
    std = train_tensor.std(dim=(0))    # 沿着 H 和 W 轴计算标准差
    
    torch.save(mean, 'global_mean.pt')
    torch.save(std, 'global_std.pt')
    np.save('global_mean.npy', mean.cpu().numpy())
    np.save('global_std.npy', std.cpu().numpy())
    print()
        
        
if __name__ == '__main__':
    H = get_sampler_hparams()
    # 创建线程来启动 Visdom 服务器
    # visdom_thread = threading.Thread(target=start_visdom_server)
    # visdom_thread.start()
    H.batch_size = 1
    # vis = set_up_visdom(H)
    print('---------------------------------')
    print(f'Setting up training for {H.sampler}')
    load(H)