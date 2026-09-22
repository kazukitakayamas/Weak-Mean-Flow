import torch

import torch.nn as nn

from models.time_sampler import sample_two_timesteps
from models.ema import init_ema, update_ema_net
from models.weak_loss import experimental_loss, weak_terms


class MeanFlow(nn.Module):
    def __init__(self, arch, args, net_configs):
        super(MeanFlow, self).__init__()
        self.net = arch(**net_configs)
        self.args = args

        # Put this in a buffer so that it gets included in the state dict
        self.register_buffer("num_updates", torch.tensor(0))
        
        self.net_ema = init_ema(self.net, arch(**net_configs), args.ema_decay)

        # maintain extra ema nets
        self.ema_decays = args.ema_decays
        for i, ema_decay in enumerate(self.ema_decays):
            self.add_module(f"net_ema{i + 1}", init_ema(self.net, arch(**net_configs), ema_decay))

    def update_ema(self):
        self.num_updates += 1
        # num_updates = self.num_updates.item()
        num_updates = self.num_updates

        update_ema_net(self.net, self.net_ema, num_updates)

        # update extra ema
        for i in range(len(self.ema_decays)):
            update_ema_net(self.net, self._modules[f"net_ema{i + 1}"], num_updates)

    def forward_with_loss(self, x, aug_cond):

        # Weak meanflow 追加
        if getattr(self.args, "method", "mf") != "mf":
            if aug_cond is not None:
                raise ValueError("Controlled experiments require aug_cond=None")
            loss, self.last_losses = experimental_loss(self.net, x, self.args)
            return loss

        device = x.device
        e = torch.randn_like(x).to(device)
        t, r = sample_two_timesteps(self.args, num_samples=x.shape[0], device=device)
        t, r = t.view(-1, 1, 1, 1), r.view(-1, 1, 1, 1)

        z = (1 - t) * x + t * e
        v = e - x

        # define network function
        def u_func(z, t, r):
            h = t - r
            return self.net(z, (t.view(-1), h.view(-1)), aug_cond)

        dtdt = torch.ones_like(t)
        drdt = torch.zeros_like(r)

        with torch.amp.autocast("cuda", enabled=False):
            u_pred, dudt = torch.func.jvp(u_func, (z, t, r), (v, dtdt, drdt))
        
            u_tgt = (v - (t - r) * dudt).detach()

            loss = (u_pred - u_tgt)**2
            loss = loss.sum(dim=(1, 2, 3))  # squared l2 loss
            
            # adaptive weighting
            adp_wt = (loss.detach() + self.args.norm_eps) ** self.args.norm_p
            loss = loss / adp_wt

            loss = loss.mean()  # mean over batch dimension
        
        return loss
    
    def loss_terms(self, x):
        """Weak method only: (diag, weak, logs) kept separate.

        The trainer backwards the two terms one after the other so the peak
        activation memory is the larger graph instead of their sum. Summing
        first and backwarding once gives identical gradients but needs both
        graphs alive at the same time.
        """
        if getattr(self.args, "method", "mf") != "weak":
            raise ValueError("loss_terms is only defined for method=weak")
        return weak_terms(self.net, x, self.args)

    @torch.no_grad()
    def sample(self, samples_shape, net=None, device=None, num_steps=1,
               generator=None, initial_noise=None, sampler="meanflow"):
        """MeanFlow transitions on a uniform decreasing time grid; NFE=num_steps."""
        if not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        if sampler not in {"meanflow", "fm_euler"}:
            raise ValueError("sampler must be meanflow or fm_euler")
        net = net if net is not None else self.net_ema
        if device is None:
            device = next(net.parameters()).device
        if initial_noise is None:
            z = torch.randn(samples_shape, dtype=torch.float32, device=device,
                            generator=generator)
        else:
            if tuple(initial_noise.shape) != tuple(samples_shape):
                raise ValueError("initial_noise shape does not match samples_shape")
            z = initial_noise.to(device=device, dtype=torch.float32).clone()
        grid = torch.linspace(1.0, 0.0, num_steps + 1, device=z.device, dtype=z.dtype)
        for i in range(num_steps):
            t = grid[i].expand(z.shape[0])
            h = (grid[i] - grid[i + 1]).expand(z.shape[0])
            # The integration step and the network's interval condition are
            # different quantities for diagonal-only Flow Matching.
            condition_h = h if sampler == "meanflow" else torch.zeros_like(h)
            u = net(z, (t, condition_h), aug_cond=None)
            z = z - h.reshape(-1, *([1] * (z.ndim - 1))) * u
        # Do not clamp intermediate states or re-inject noise.
        return z
