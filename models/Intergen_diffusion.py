import torch
from torch import nn
from models.intergen.utils import *
from models.intergen.blocks import *
from models.intergen.intergen_utils import *

from models.intergen.gaussian_diffusion import (
    MotionDiffusion,
    space_timesteps,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)

class InterDenoiser(nn.Module):
    def __init__(self,
                 input_feats, # 262
                 latent_dim=512,
                 num_frames=240,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 activation="gelu",
                 **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim

        self.text_emb_dim = 768

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Input Embedding
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(TransformerBlock(num_heads=num_heads,latent_dim=latent_dim, dropout=dropout, ff_size=ff_size))
        # Output Module
        self.out = zero_module(FinalLayer(self.latent_dim, self.input_feats))


    def forward(self, x, timesteps, mask=None, cond=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]
        x_a = x[...,:self.input_feats]

        if mask is not None:
            mask = mask[...,0]

        emb = self.embed_timestep(timesteps)

        a_emb = self.motion_embed(x_a)
        h_a = self.sequence_pos_encoder(a_emb) # h_a^0 in FIg. 5 of InterGen paper, i = 0

        if mask is None:
            mask = torch.ones(B, T).to(x_a.device)
        key_padding_mask = ~(mask > 0.5)

        for i,block in enumerate(self.blocks):
            h_a = block(h_a, emb, key_padding_mask)

        output = self.out(h_a)

        return output

class InterDiffusion(nn.Module):
    def __init__(self, H, embedding_weight):
        super().__init__()
        self.nfeats = 256
        self.latent_dim = 256
        self.ff_size = 1024
        self.num_layers = 8
        self.num_heads = 8
        self.dropout = 0.1
        self.activation = 'gelu'
        self.motion_rep = 'global'
        self.T_BAR = 700
        self.diffusion_steps = 1000
        self.beta_scheduler = 'cosine'
        self.sampler = 'uniform'
        self.sampling_strategy = 'ddim50'
        self.normalizer = MotionNormalizer()
        
        self.net = InterDenoiser(self.nfeats, self.latent_dim, ff_size=self.ff_size, num_layers=self.num_layers,
                                       num_heads=self.num_heads, dropout=self.dropout, activation=self.activation)

        self.betas = get_named_beta_schedule(self.beta_scheduler, self.diffusion_steps)

        timestep_respacing=[self.diffusion_steps]
        self.diffusion = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
        )
        self.sampler = create_named_schedule_sampler(self.sampler, self.diffusion)
        
        self.n_samples = H.batch_size
        self.codebook_size = H.codebook_size
        self.embed_dim = 256
        self.latent_shape = [16, 16]
        self.embedding_weight = embedding_weight
        

    def latent_ids_to_onehot(self, latent_ids):
        min_encoding_indices = latent_ids.view(-1).unsqueeze(1)
        encodings = torch.zeros(
            min_encoding_indices.shape[0],
            self.codebook_size
        ).to(latent_ids.device)
        encodings.scatter_(1, min_encoding_indices, 1)
        one_hot = encodings.view(
            latent_ids.shape[0],
            self.latent_shape[0],
            self.latent_shape[1],
            self.codebook_size
        )
        return one_hot.reshape(one_hot.shape[0], -1, self.codebook_size)

    def embed(self, z):
        with torch.no_grad():
            z_flattened = z.view(-1, self.codebook_size)  # B*H*W, codebook_size
            embedded = torch.matmul(z_flattened, self.embedding_weight).view(
                z.size(0),
                self.latent_shape[0],
                self.latent_shape[1],
                self.embed_dim
            ).permute(0, 3, 1, 2).contiguous()

        return embedded

    def compute_loss(self, batch):
        idx = batch
        B, C = idx.shape
        latents_one_hot = self.latent_ids_to_onehot(idx)
        x_start = self.embed(latents_one_hot).view(B, C, -1) # [b, embed_dim]
        
        cond = None    # None
        seq_mask = None # used for variable-length sequences
        # cond = batch['cond']
        # seq_mask = self.generate_src_mask(batch["motions"].shape[1], batch["motion_lens"]).to(x_start.device)
        
        t, _ = self.sampler.sample(B, x_start.device)
        output = self.diffusion.training_losses(
            model=self.net,
            x_start=x_start,
            t=t,
            mask=seq_mask,
            t_bar=self.T_BAR,
            model_kwargs={"mask":seq_mask,
                          "cond":cond,
                          },
        )
        return output   # {"total"}

    def forward(self, batch):
        zs_t = batch
        B, C, D = zs_t.shape
        
        timestep_respacing= self.sampling_strategy
        self.diffusion_test = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
        )

        output = self.diffusion_test.ddim_sample_loop(
            self.net,
            (B, C, self.nfeats),
            clip_denoised=False,
            progress=True,
            model_kwargs={
                "mask":None,
                "cond":None,
            },
            x_start=None)
        return {"output":output}
    
    def train_iter(self, x):
        loss = self.compute_loss(x)
        stats = {'loss': loss['total']}
        return stats
    
    def random_discrete_sample(self):
        idx_t = torch.randint(0, self.codebook_size, (self.n_samples, self.latent_shape[0], self.latent_shape[1])).to(self.embedding_weight.device)
        latents_one_hot = self.latent_ids_to_onehot(idx_t)
        B, C, _ = latents_one_hot.shape
        zs_t = self.embed(latents_one_hot).view(B, C, -1)
        zs_hat_0 = self(zs_t)['output']
        zs_hat_0 = self.normalizer.backward(zs_hat_0.detach())
        return zs_hat_0.reshape(B, C, self.latent_shape[0], self.latent_shape[1])
    
    def random_continuous_sample(self):
        zs_t = torch.randn(self.n_samples, np.prod(self.latent_shape), self.embed_dim).cuda(0)
        B, C, _ = zs_t.shape
        zs_hat_0 = self(zs_t)['output']
        zs_hat_0 = self.normalizer.backward(zs_hat_0.detach())
        return zs_hat_0.reshape(B, C, self.latent_shape[0], self.latent_shape[1])
    
    def sample(self):
        discrete_sample = self.random_discrete_sample()
        continuous_sample = self.random_continuous_sample()
        
        # return torch.cat([discrete_sample, continuous_sample], dim=-1)
        return discrete_sample