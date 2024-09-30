import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformers.tisa_transformer import TisaTransformer
from math import sqrt
import numpy as np

class Conv1dLayer(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1):
    super().__init__()
    self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
    nn.init.kaiming_normal_(self.conv1d.weight)

  def forward(self, x):
    return self.conv1d(x.permute(0,2,1)).permute(0,2,1)

def silu(x):
    return x * torch.sigmoid(x)

class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps, in_channels, hidden_channels):
        super().__init__()
    
        self.in_channels = in_channels
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = nn.Linear(in_channels, hidden_channels)
        self.projection2 = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)          # [1,64]
        table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table
    
class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, 
                embedding_dim, 
                nn_name,
                nn_args,
                index):

        super().__init__()
        if nn_name=="tisa":
            dilation_cycle = nn_args["dilation_cycle"]
            dilation=dilation_cycle[(index % len(dilation_cycle))]
            self.nn = TisaTransformer(residual_channels, 2 * residual_channels, d_model=residual_channels, num_blocks=nn_args["num_blocks"], num_heads=nn_args["num_heads"], activation=nn_args["activation"], norm=nn_args["norm"], drop_prob=nn_args["dropout"], d_ff=nn_args["d_ff"], seqlen=nn_args["seq_len"], use_preln=nn_args["use_preln"], bias=nn_args["bias"], dilation=dilation)
        elif nn_name=="conv":
            dilation=2**(index % nn_args["dilation_cycle_length"])
            self.nn = Conv1dLayer(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        else:
            raise ValueError(f"Unknown nn_name: {nn_name}")
            
        
        self.diffusion_projection = nn.Linear(embedding_dim, residual_channels)
        self.output_projection = Conv1dLayer(residual_channels, 2 * residual_channels, 1)
        self.residual_channels = residual_channels

    def forward(self, x, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(1)
        y = x + diffusion_step # [96, 150, 256], (B, T, C)

        y = self.nn(y).squeeze(-1)

        gate, filter = torch.chunk(y, 2, dim=2)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=2)
        return (x + residual) / sqrt(2.0), skip


class LDA(nn.Module):
    def __init__(self,
                pose_dim,
                residual_layers,
                residual_channels,
                embedding_dim,
                n_noise_schedule,
                nn_name,
                nn_args):
        super().__init__()
        self.input_projection = Conv1dLayer(pose_dim, residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(n_noise_schedule, 128, embedding_dim)

        self.residual_layers = nn.ModuleList([
            ResidualBlock(residual_channels,
                embedding_dim,
                nn_name,
                nn_args,
                i)
            for i in range(residual_layers)
        ])
        self.skip_projection = Conv1dLayer(residual_channels, residual_channels, 1)
        self.output_projection = Conv1dLayer(residual_channels, pose_dim, 1)
        nn.init.zeros_(self.output_projection.conv1d.weight)

    def forward(self, x, diffusion_step):
        '''
        x: noisy zs
        '''
        # input mapping
        x = self.input_projection(x)
        x = F.relu(x)
        
        # get diffusion timestamps
        diffusion_step = self.diffusion_embedding(diffusion_step)

        skip = None

        # denoise
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step)
            skip = skip_connection if skip is None else skip_connection + skip
        if skip is not None:
            x = skip / sqrt(len(self.residual_layers))
        
        # output mapping
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x


class LDA_Diffusion(nn.Module):
    def __init__(self, H, embedding_weight, generator):
        super().__init__()
        
        # self.input_dim =       # image channels
        self.image_size = 256
        self.embed_dim = 256    # embedding dimension
        self.unconditional = True
        
        # dict
        diff_params = {'residual_layers': 20, 
                       'residual_channels': 256, 
                       'embedding_dim': 512, 
                       'nn_name': 'tisa', 
                       'nn_args': {'num_blocks': 2, 
                                   'num_heads': 8, 
                                   'activation': 'relu', 
                                   'dropout': 0.1, 
                                   'norm': 'LN', 
                                   'd_ff': 1024, 
                                   'seq_len': 150, 
                                   'use_preln': 'false', 
                                   'bias': 'false', 
                                   'dilation_cycle': [0, 1, 2]
                                   }
                       }
            
        beta_min = 0.01
        beta_max = 0.7
        self.n_noise_schedule = 150
        
        self.noise_schedule_name = "linear"                                                         
        self.noise_schedule = torch.linspace(beta_min, beta_max, self.n_noise_schedule)
        self.noise_level = torch.cumprod(1 - self.noise_schedule, dim=0)
        
        nn_name = 'tisa'
        # dict
        nn_args = {"num_blocks": 2, 
                   "num_heads": 8, 
                   "activation": 'relu', 
                   "dropout": 0.1, 
                   "norm": 'LN', 
                   "d_ff": 1024, 
                   "seq_len": 150, 
                   "use_preln": 'false', 
                   "bias": 'false', 
                   "dilation_cycle": [0,1,2]
                   }
        
        self.diffusion_model = LDA(self.embed_dim, 
                                        diff_params["residual_layers"],
                                        diff_params["residual_channels"],
                                        diff_params["embedding_dim"],
                                        self.n_noise_schedule,
                                        nn_name,
                                        nn_args)
        self.loss_fn = nn.MSELoss()
        
        self.embedding = embedding_weight
        self.generator = generator

    def get_codebook_entry(self, idx):
        return self.embedding[idx]
    
    # @torch.compile(mode='max-autotune', fullgraph=True) # useful
    def diffusion(self, zs, t):
        N, T, C = zs.shape
        noise = torch.randn_like(zs)
        noise_scale = self.noise_level.type_as(noise)[t].unsqueeze(1).unsqueeze(2).repeat(1,T,C)
        noise_scale_sqrt = noise_scale**0.5
        noisy_zs = noise_scale_sqrt * zs + (1.0 - noise_scale)**0.5 * noise
        return noisy_zs, noise
        
    # @torch.compile(mode='max-autotune', fullgraph=True) # useful
    def forward(self, batch):
        idx = batch # [b, 256]
        zs = self.get_codebook_entry(idx) # [b, embed_dim]
        N, C, D = zs.shape

        num_noisesteps = self.n_noise_schedule
        t = torch.randint(0, num_noisesteps, [N], device=zs.device)

        noisy_zs, noise = self.diffusion(zs, t)
        predicted = self.diffusion_model(noisy_zs, t)

        loss = self.loss_fn(noise, predicted.squeeze(1))
                
        return loss
    
    def train_iter(self, x):
        loss = self(x)
        stats = {'loss': loss}
        return stats
    
    def sample(self):
        noise_sched_0 = self.noise_schedule
        beta = np.array(noise_sched_0)

        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)
        T = np.arange(0,len(beta), dtype=np.float32)

        zs = torch.randn(1, 1, self.embed_dim).cuda(0)
                
        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n]**0.5
            c2 = beta[n] / (1 - alpha_cum[n])**0.5
                
            diff = self.diffusion_model(zs, torch.tensor([T[n]], device=zs.device)).squeeze(1)
            
            zs = c1 * (zs - c2 * diff)
                
            if n > 0:
                noise = torch.randn_like(zs)
                sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                zs += sigma * noise
                
        predected_zs_start = zs.cpu().detach().numpy()
        return predected_zs_start