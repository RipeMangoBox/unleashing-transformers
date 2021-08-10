import torch
import torch.nn.functional as F
import torch.distributions as dists
import numpy as np
import math
import random
from .sampler import Sampler


class AbsorbingDiffusion(Sampler):
    def __init__(self, H, denoise_fn, mask_id, embedding_weight, aux_weight=0.01):
        super().__init__(H, embedding_weight=embedding_weight)

        self.num_classes = H.codebook_size
        self.latent_emb_dim = H.emb_dim
        self.shape = tuple(H.latent_shape)
        self.num_timesteps = H.diffusion_steps
        self.mask_id = mask_id
        self._denoise_fn = denoise_fn
        self.n_samples = min(H.batch_size, 64)
        self.loss_type = H.loss_type
        self.mask_schedule = H.mask_schedule
        self.aux_weight = aux_weight
        self.deepspeed = H.deepspeed
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps+1))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps+1))
        self.register_buffer('loss_history', torch.zeros(self.num_timesteps+1))

        assert self.mask_schedule in ['random', 'fixed']
    
    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            # why do we square just to sqrt again!?!?
            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(1, self.num_timesteps+1, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt

        # try cosine noise schedule
        else:
            raise ValueError
    
    def q_sample(self, x_0, t):
        # randomly set token to mask with probability t/T
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        mask = torch.rand_like(x_t.float()) < (t.float().unsqueeze(-1) / self.num_timesteps)
        x_t[mask] = self.mask_id
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask
    
    def q_sample_mlm(self, x_0, t):
        # testing fixed noise schedule
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        mask = torch.zeros_like(x_t).to(torch.bool)

        # TODO: offset so each n_masked_tokens is picked with equal probability
        n_masked_tokens = (t.float() / self.num_timesteps) * x_t.size(1)
        n_masked_tokens = torch.round(n_masked_tokens).to(torch.int64)
        n_masked_tokens[n_masked_tokens == 0] = 1
        ones = torch.ones_like(mask[0]).to(torch.bool)

        for idx, n_tokens_to_mask in enumerate(n_masked_tokens):
            index = torch.randperm(x_0.size(1))[:n_tokens_to_mask].to(x_0.device)
            mask[idx].scatter_(dim=0, index=index, src=ones)
        
        x_t[mask] = self.mask_id
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask

    def _train_loss(self, x_0):
        b, device = x_0.size(0), x_0.device

        # choose what time steps to compute loss at
        t, pt = self.sample_time(b, device, 'uniform')

        # make x noisy and denoise

        if self.mask_schedule == 'random':
            x_t, x_0_ignore, mask = self.q_sample(x_0=x_0, t=t)
        elif self.mask_schedule == 'fixed':
            x_t, x_0_ignore, mask = self.q_sample_mlm(x_0=x_0, t=t)

            
        x_0_hat_logits = self._denoise_fn(x_t, t=t).permute(0,2,1)

        # Always compute ELBO for comparison purposes
        vb_loss = F.cross_entropy(x_0_hat_logits, x_0_ignore, ignore_index=-1, reduction='none').sum(1) / t
        vb_loss = vb_loss / pt
        vb_loss = vb_loss / (math.log(2) * x_0.shape[1:].numel())        
        
        if self.loss_type == 'elbo':
            loss = vb_loss 
        elif self.loss_type == 'd3pm':
            # a reweighted variational bound that also takes into account non-masked points
            aux_loss = F.cross_entropy(x_0_hat_logits, x_0, reduction='none').mean(1)
            # not sure about aux_weighting. BUG: vb_loss is now normed (i.e. / pt) so weighting will be even more broken
            loss = vb_loss + self.aux_weight * aux_loss
        elif self.loss_type == 'normed':
            denom = mask.float().sum(1)
            denom[denom == 0] = 1 # prevent divide by 0 errors.
            # print(denom)
            # print((F.cross_entropy(x_0_hat_logits, x_0_ignore, ignore_index=-1, reduction='none') != 0).float().sum(1))
            # print("---------------------------")
            loss = F.cross_entropy(x_0_hat_logits, x_0_ignore, ignore_index=-1, reduction='none').sum(1) / denom
        else:
            raise ValueError

        # Track loss at each time step history for bar plot
        Lt2_prev = self.loss_history.gather(dim=0, index=t)
        new_loss_history = (0.1 * loss + 0.9 * Lt2_prev).detach()
        if self.deepspeed:
            self.loss_history.scatter_(dim=0, index=t, src=new_loss_history.half())
        else:
            self.loss_history.scatter_(dim=0, index=t, src=new_loss_history)

        # Track loss at each time step for importance sampling
        Lt2 = vb_loss.detach().clone().float().pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        if self.deepspeed:
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history.half())
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2).half())
        else:
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        return loss.mean(), vb_loss.mean()
    
    def sample(self, num_steps):
        b, device = self.n_samples, 'cuda'
        x_0 = torch.ones((b, np.prod(self.shape)), device=device).long() * self.mask_id
        # endpoint = False since the last step often makes no changes
        time_steps = np.linspace(self.num_timesteps, 1, num=num_steps, endpoint=False, dtype=np.int)
        for t in time_steps:
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)
            x_t, _, _ = self.q_sample(x_0, t)
            x_0_logits = self._denoise_fn(x_t, t=t)
            x_0_dist = dists.Categorical(
                logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_0[x_t == self.mask_id] = x_0_hat[x_t == self.mask_id]
            # print("x0 at", t, x_0, x_0.shape)

        return x_0
    
    def sample_shape(self, shape, num_samples, num_steps=1000):
        x_0 = torch.ones((num_samples,) + shape, device='cuda').long() * self.mask_id
        time_steps = np.linspace(self.num_timesteps, 1, num=num_steps, endpoint=False, dtype=np.int)
        x_lim, y_lim = shape[0] - self.shape[1], shape[1] - self.shape[2]

        for t in time_steps:
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((num_samples,), t, device='cuda', dtype=torch.long)

            # make x_0 noisy
            x_t, _, mask = self.q_sample(x_0.reshape(x_0.size(0), -1), t)
            x_t = x_t.reshape(x_t.size(0), shape[0], shape[1])
            mask = mask.reshape(x_t.size(0), shape[0], shape[1])
            # keep track of PoE probabilities
            x_0_probs = torch.zeros((num_samples,) + shape + (self.codebook_size,), device='cuda')
            # keep track of counts
            count = torch.zeros((num_samples,) + shape, device='cuda')

            # TODO: Use step size > 1
            for i in range(x_lim+1):
                for j in range(y_lim+1):
                    # collect local noisy area
                    x_t_part = x_t[:,i:i+self.shape[1], j:j+self.shape[2]]

                    # increment count
                    count[:,i:i+self.shape[1], j:j+self.shape[2]] += 1.0

                    # flatten
                    x_t_part = x_t_part.reshape(x_t_part.size(0), -1)
                    
                    # denoise
                    x_0_logits_part = self._denoise_fn(x_t_part, t=t)

                    # unflatten
                    x_0_logits_part = x_0_logits_part.reshape(x_t_part.size(0), self.shape[1], self.shape[2], -1)

                    # multiply probabilities
                    # for mixture
                    x_0_probs[:,i:i+self.shape[1], j:j+self.shape[2]] += torch.softmax(x_0_logits_part, dim=-1)
                    # for PoE
                    # x_0_probs[:,i:i+self.shape[1], j:j+self.shape[2]] += torch.log_softmax(x_0_logits_part, dim=-1)

            # Normalize probabilities
            
            # Product of Experts -ish (with count division probably same as a mixture)
            # temp = 4.0
            # x_0_probs = torch.softmax(x_0_probs / count.unsqueeze(-1), dim=-1)

            # Mixture with Temperature
            x_0_probs = x_0_probs / x_0_probs.sum(-1, keepdim=True)
            temp = 0.92
            C = torch.tensor(x_0_probs.size(-1)).float()
            x_0_probs = torch.softmax((torch.log(x_0_probs) + torch.log(C)) / temp, dim=-1)

            x_0_dist = dists.Categorical(probs=x_0_probs)
            x_0_hat = x_0_dist.sample().long()

            # update x_0 where anything has been masked
            x_0[mask] = x_0_hat[mask]

        return x_0


    def train_iter(self, x):
        loss, vb_loss = self._train_loss(x)
        stats = {'loss': loss, 'vb_loss': vb_loss}
        return stats

# NOTE: Is there any point spending as long reducing the loss on the later time 
#       steps when it can't be improved easily? Maybe that's why it's /t