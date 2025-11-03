import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyLearning(nn.Module):
    """
    Pipeline:
      1) encode person image x_p -> z_0
      2) sample noise epsilon ~ N(0, 1)
      3) build noisy latent: z_t = (1 - t) * z_0 + t * epsilon
      4) predict conditional noise: eps_ct = DenoisingDiT(z_t, t, cond)
      5) reconstruct clean latent: z0_hat = (z_t - t * eps_ct) / (1 - t)
      6) decode to pixel: x_tr_hat = VAE.decode(z0_hat)
      7) compute frequency loss: Lfreq = ||F(x_tr_hat ⊙ m_g) - F(x_p ⊙ m_g)||^2
      8) return total loss = denoise_loss + lambda_f * freq_loss
                           = E[w(t) * ||eps_ct - eps||^2] + λ * ||F(x_tr_hat ⊙ m_g) - F(x_p ⊙ m_g)||^2
    """
    def __init__(self, vae, dit, lambda_freq: float = 0.1):
        """
        Args:
            vae: VAE with .encode(x) and .decode(z)
            dit: DenoisingDiT 
            lambda_freq: weight for frequency loss
        """
        super().__init__()
        self.vae = vae
        self.dit = dit
        self.lambda_freq = lambda_freq
        self.scale = getattr(self.vae.config, "scaling_factor", 1.0)
        self.shift = getattr(self.vae.config, "shift_factor", 0.0)

    def _vae_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image x to latent z_0 using VAE encoder.
        """
        encoded = self.vae.encode(x)
        if hasattr(encoded, "latent_dist"):
            z_0 = encoded.latent_dist.mode()
        else:
            z_0 = encoded

        z_0 = (z_0 - self.shift) * self.scale   # normalization
        return z_0


    def _call_dit(self, z_t: torch.Tensor, t_cont: torch.Tensor, cond: dict) -> torch.Tensor:
        """
        Args:
            z_t: [B, Cz, H', W']
            t_cont: [B,1,1,1] continuous t (0,1)
            cond: condition dict for DiT
        Returns:
            eps_ct: predicted noise, same shape as z_t
        """
        B = z_t.shape[0]
        t_int = (t_cont.view(B) * 999).long()  # [B]  timestep is LongTensor

        encoder_hidden_states = cond.get("encoder_hidden_states", None)
        pooled_projections = cond.get("pooled_projections", None)
        ref_key = cond.get("ref_key", None)
        ref_value = cond.get("ref_value", None)
        pose_cond = cond.get("pose_cond", None)
        block_controlnet_hidden_states = cond.get("block_controlnet_hidden_states", None)
        joint_attention_kwargs = cond.get("joint_attention_kwargs", None)

        # In FitDiT paper, eps_ct = DiT(z_t; Ivec, t)
        output = self.dit(
            hidden_states=z_t,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=t_int,
            block_controlnet_hidden_states=block_controlnet_hidden_states,
            joint_attention_kwargs=joint_attention_kwargs,
            ref_key=ref_key,
            ref_value=ref_value,
            pose_cond=pose_cond,
            return_dict=True,
        )

        if hasattr(output, "sample"):
            eps_ct = output.sample
        else:
            eps_ct = output
        return eps_ct


    def forward(
        self,
        x_person: torch.Tensor,   # [B,3,H,W] person image
        x_garment: torch.Tensor,  # [B,3,H,W] garment image
        m_garment: torch.Tensor,  # [B,1,H,W] or [B,H,W] mask in image space
        cond: dict = None,  
    ):
        B, C, H, W = x_person.shape
        device = x_person.device

        # ensure mask shape
        if m_garment.dim() == 3:
            m_garment = m_garment.unsqueeze(1)  # [B,1,H,W]

        # use VAE to encode person & garment to latent space
        z_person = self._vae_encode(x_person)     # [B, Cz, H', W']
        # z_garment = self._vae_encode(x_garment)   # [B, Cz, H', W']
        _, Cz, H_lat, W_lat = z_person.shape

        # downsample mask to latent 
        # mask_latent = F.interpolate(m_garment, size=(H_lat, W_lat), mode="nearest")   

        # sample noise and timestep(z_person or z_garment for following steps???)
        eps = torch.randn_like(z_person)    # standard normal distribution
    
        # expand to latent shape 
        t = torch.rand(B, 1, 1, 1, device=device)   # [B,1,1,1]
        t_lat = t.expand(B, Cz, H_lat, W_lat)     # [B,Cz,H',W']

        # z_t = (1 - t) * z_0 + t * eps
        z_t = (1.0 - t_lat) * z_person + t_lat * eps

        # predict conditional noise eps_ct = f_θ(z_t, t, cond)
        eps_ct = self._call_dit(z_t, t, cond)     # same shape as z_t

        # latent denoising loss (!! Need to multiply the weighting function w(t) in paper)
        loss_denoise = F.mse_loss(eps_ct, eps)

        # estimate clean latent z0_hat = (z_t - t * eps_ct) / (1 - t)       
        denom = (1.0 - t_lat).clamp(min=1e-4)  # t in latent shape
        z0_hat = (z_t - t_lat * eps_ct) / denom

        # decode to pixel space → x_tr_hat  (predicted try-on)
        # x_tr_hat = x_person # [For test, then the loss_freq will is 0]
        x_tr_hat = self.vae.decode(z0_hat)        # [B,3,H,W] 

        # frequency loss in IMAGE space L_f = ||F(x_tr_hat ⊙ m_g) - F(x_p ⊙ m_g)||^2
        x_tr_masked = x_tr_hat * m_garment
        x_p_masked = x_person * m_garment

        F_tr = torch.fft.fftn(x_tr_masked, dim=(-2, -1))
        F_p  = torch.fft.fftn(x_p_masked,  dim=(-2, -1))
        F_diff = F_tr - F_p
        F_dist_sq = F_diff.real**2 + F_diff.imag**2
        loss_freq = F_dist_sq.mean()

        loss_total = loss_denoise + self.lambda_freq * loss_freq

        return loss_total, loss_denoise, loss_freq, x_tr_hat
