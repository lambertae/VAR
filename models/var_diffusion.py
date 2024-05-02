import torch
import torch.nn as nn


from dit.latentmlp import SimpleMLP
from dit.latentmlp_adaln import SimpleMLPAdaLN
from dit.diffusion import create_diffusion
from dit.simpledit import DiT_models
from dit.unet import UNetModel
class Diffusion(nn.Module):
    
    def __init__(self, use_dit=False,
                 simplemlp_d=12,
                 simplemlp_w=1536,
                 use_adaln=False,
                 gen_schedule='cosine',
                 num_sampling_steps='250',
                 cond_drop=0.0,
                 pad_with_mask_token=True,
                 diffusion_batch_mul=1,
                 rdm_type="simplemlp",
                 noise_schedule="linear",
                 finetune=False,
                 mage_adaln=False,
                 direct_pred=False,
                 beta_start=0.0001,
                 beta_end=0.02, 
                 decoder_embed_dim=32,
                 channels = 32, 
                 diffusion_dropout=0.0, 
                 pred_mode="eps"):
        super().__init__()
        self.channels = channels
        self.direct_pred = direct_pred
        self.diffusion_batch_mul = diffusion_batch_mul
        
        if use_dit:
            self.rdm = DiT_models["DiT-S"](input_size=2, context_dim=decoder_embed_dim, in_channels=self.channels)
        else:
            if use_adaln:
                self.rdm = SimpleMLPAdaLN(
                    in_channels=self.channels,
                    model_channels=simplemlp_w,
                    out_channels=self.channels,
                    num_res_blocks=simplemlp_d,
                    context_channels=decoder_embed_dim,
                    cond_drop=cond_drop,
                    dropout=diffusion_dropout
                )
            else:
                self.rdm = SimpleMLP(
                    in_channels=self.channels,
                    time_embed_dim=simplemlp_w,
                    model_channels=simplemlp_w,
                    bottleneck_channels=simplemlp_w,
                    out_channels=self.channels,
                    num_res_blocks=simplemlp_d,
                    dropout=0,
                    use_context=True,
                    context_channels=decoder_embed_dim
                )

        self.initialize_diffusion(pred_mode, num_sampling_steps, noise_schedule, beta_start, beta_end)
        
    
    def initialize_diffusion(self, pred_mode, num_sampling_steps, noise_schedule, beta_start, beta_end, use_ddim=None):
        self.num_sampling_steps = num_sampling_steps
        if use_ddim is None:
            self.use_ddim = "ddim" in num_sampling_steps
        else:
            self.use_ddim = use_ddim
        if pred_mode == "x0":
            self.train_diffusion = create_diffusion(timestep_respacing="", predict_xstart=True,
                                                    beta_start=beta_start, beta_end=beta_end)
            self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, predict_xstart=True,
                                                  beta_start=beta_start, beta_end=beta_end)
        elif pred_mode == "eps":
            self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule=noise_schedule,
                                                    predict_xstart=False, beta_start=beta_start, beta_end=beta_end)
            self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule=noise_schedule,
                                                  predict_xstart=False, beta_start=beta_start, beta_end=beta_end)
        else:
            raise NotImplementedError

    def forward_diffusion(self, z, cond, mask):
        bsz, seq_len, _ = z.shape
        z = z.reshape(bsz*seq_len, -1)
        cond = cond.reshape(bsz*seq_len, -1)
        mask = mask[:, self.buffer_size:].reshape(bsz*seq_len)

        z = z.repeat(self.diffusion_batch_mul, 1)
        cond = cond.repeat(self.diffusion_batch_mul, 1)
        mask = mask.repeat(self.diffusion_batch_mul)

        if self.direct_pred:
            pred_z = self.cond_proj(cond)
            loss = nn.functional.mse_loss(pred_z, z, reduction='none').mean(-1)
        else:
            t = torch.randint(0, self.train_diffusion.num_timesteps, (bsz*seq_len*self.diffusion_batch_mul,)).cuda()
            model_kwargs = dict(c=cond)
            loss_dict = self.train_diffusion.training_losses(self.rdm, z, t, model_kwargs)
            loss = loss_dict["loss"]
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def sampler(self, cond):
        z = torch.randn(cond.shape[0], self.channels).cuda()
        model_kwargs = dict(c=cond)
        sample_fn = self.rdm.forward

        if self.use_ddim:
            sampled_token_latent = self.gen_diffusion.ddim_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False
            )
        else:
            sampled_token_latent = self.gen_diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False
            )
        
        return sampled_token_latent