"""
Unbalanced Optimal Transport (UOT) cross-modal fusion for MMA-DFER.

The baseline model fuses audio and video by mean-pooling one modality and
broadcasting it as a global bias onto every token of the other one
(see models/Generate_Model.py, inside the block loop). That treats every
audio frame as equally relevant to every video frame.

Here we instead solve a small entropic *unbalanced* OT problem between the
16 video frame descriptors and the 32 audio time slots, and use the transport
plan to route information across modalities. Unbalanced (KL-relaxed marginals)
matters for in-the-wild DFER: silence, background music and off-screen speech
have no visual counterpart, so the plan is allowed to create/destroy mass
instead of being forced to match everything.
"""

import math

import torch
from torch import nn


def unbalanced_sinkhorn_log(C, log_a=None, log_b=None, eps=0.05, tau=1.0, n_iters=10):
    """Log-domain scaling iterations for entropic UOT with KL marginal penalties.

    Solves    min_pi  <C, pi> + eps*KL(pi | a x b) + tau*KL(pi 1 | a) + tau*KL(pi^T 1 | b)

    Args:
        C: cost matrix, [B, N, M]
        log_a, log_b: log of the source/target measures, [B, N] / [B, M].
            Defaults to uniform.
        eps: entropic regularization. Smaller -> sharper (more one-to-one) plan.
        tau: marginal relaxation. tau -> inf recovers balanced Sinkhorn,
            small tau lets the plan drop unmatched tokens.
        n_iters: number of scaling iterations (unrolled, differentiable).

    Returns:
        pi: transport plan, [B, N, M]. Its total mass is <= 1 and is *not*
            normalized on purpose: the mass is the "how much audio evidence
            does this frame have" signal we want downstream.
    """
    B, N, M = C.shape
    if log_a is None:
        log_a = C.new_full((B, N), -math.log(N))
    if log_b is None:
        log_b = C.new_full((B, M), -math.log(M))

    f = C.new_zeros(B, N)
    g = C.new_zeros(B, M)
    scale = tau / (tau + eps)

    for _ in range(n_iters):
        f = -scale * eps * torch.logsumexp((g.unsqueeze(1) - C) / eps + log_b.unsqueeze(1), dim=2)
        g = -scale * eps * torch.logsumexp((f.unsqueeze(2) - C) / eps + log_a.unsqueeze(2), dim=1)

    log_pi = (f.unsqueeze(2) + g.unsqueeze(1) - C) / eps + log_a.unsqueeze(2) + log_b.unsqueeze(1)
    return log_pi.exp()


class UOTFusion(nn.Module):
    """One UOT fusion step, meant to be instantiated once per transformer block.

    Operates in the 128-d latent space that MMA-DFER already uses for
    cross-modal exchange, so it is cheap: the OT problem is 16 x 32 per sample.

    Both output gates are zero-initialized (tanh(0) = 0), so at step 0 the model
    is bit-identical to the baseline and training starts from a known-good point.
    """

    def __init__(self, dim=128, n_frames=16, n_image=196, n_audio_t=32, n_audio_f=8,
                 eps=0.05, tau=1.0, n_iters=10, detach_plan=False, video_token='cls'):
        super().__init__()
        self.dim = dim
        self.n_frames = n_frames
        self.n_image = n_image
        self.n_audio_t = n_audio_t
        self.n_audio_f = n_audio_f
        self.n_iters = n_iters
        self.detach_plan = detach_plan
        self.video_token = video_token

        self.eps = eps
        self.tau = tau

        self.norm_v = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)
        self.proj_a2v = nn.Linear(dim, dim)
        self.proj_v2a = nn.Linear(dim, dim)

        self.gate_a2v = nn.Parameter(torch.zeros(1))
        self.gate_v2a = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.proj_a2v.bias)
        nn.init.zeros_(self.proj_v2a.bias)

    def _video_tokens(self, image_lowdim, n, t):
        """[B, L_img, D] -> [n, t, D], one descriptor per frame."""
        x = image_lowdim.view(n, t, -1, self.dim)
        if self.video_token == 'cls':
            return x[:, :, 0, :]
        return x[:, :, 1:1 + self.n_image, :].mean(dim=2)

    def _audio_tokens(self, audio_lowdim):
        """[n, 1 + n_audio + n_prompt, D] -> [n, n_audio_t, D], pooled over frequency."""
        n_patch = self.n_audio_t * self.n_audio_f
        patches = audio_lowdim[:, 1:1 + n_patch, :]
        return patches.view(-1, self.n_audio_t, self.n_audio_f, self.dim).mean(dim=2)

    def forward(self, image_lowdim, audio_lowdim, n, t):
        """Returns (a2v, v2a) residuals to be added to the two latent streams.

        a2v: [n * t, 1, D]  -- audio evidence aligned to each video frame
        v2a: [n, 1, D]      -- video evidence summarized for the audio stream
        """
        v = self._video_tokens(image_lowdim, n, t)   # [n, t, D]
        a = self._audio_tokens(audio_lowdim)         # [n, n_audio_t, D]

        v_n = torch.nn.functional.normalize(self.norm_v(v), dim=-1)
        a_n = torch.nn.functional.normalize(self.norm_a(a), dim=-1)

        C = 1.0 - torch.bmm(v_n, a_n.transpose(1, 2))          # [n, t, n_audio_t]
        pi = unbalanced_sinkhorn_log(C, eps=self.eps, tau=self.tau, n_iters=self.n_iters)
        if self.detach_plan:
            pi = pi.detach()

        # Row-normalized plan = soft alignment; row mass = how much audio the
        # frame actually matched (this is what "unbalanced" buys us).
        row_mass = pi.sum(dim=2, keepdim=True)                  # [n, t, 1]
        col_mass = pi.sum(dim=1, keepdim=True).transpose(1, 2)  # [n, n_audio_t, 1]

        a2v = torch.bmm(pi / row_mass.clamp_min(1e-8), a)                       # [n, t, D]
        v2a = torch.bmm(pi.transpose(1, 2) / col_mass.clamp_min(1e-8), v)       # [n, n_audio_t, D]

        # Rescale by relative matched mass: frames with no audio counterpart get
        # a smaller update instead of being forced to absorb something.
        a2v = a2v * (row_mass * self.n_frames)
        v2a = v2a * (col_mass * self.n_audio_t)

        a2v = torch.tanh(self.gate_a2v) * self.proj_a2v(a2v)
        v2a = torch.tanh(self.gate_v2a) * self.proj_v2a(v2a)

        # Back to the shapes the two streams expect (broadcast over tokens).
        a2v = a2v.reshape(n * t, 1, self.dim)
        v2a = v2a.mean(dim=1, keepdim=True)

        return a2v, v2a
