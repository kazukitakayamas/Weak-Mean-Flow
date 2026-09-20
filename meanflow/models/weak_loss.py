"""Weak-form MeanFlow objective.

Appendix references (MeanFlow_Mathematical_Appendix_JA):

  D.34   uniform-triangle sampler q0(r,t) = 2 on 0 < r < t < 1
  D.43   test function phi_m = (1-t) cos(xi_m)          ("vanishing" family)
  D.55   A_y phi_m = -cos(xi) - (1-t) sin(xi) (c_m + a_m . y)
  D.75   U statistic over distinct pairs
  D.99   O(B M d) matrix form of the minibatch loss
  D.115  diagonal (flow-matching) loss, normalised per coordinate
  D.116  L = L_diag + lambda L_weak
  D.121  importance correction for a non-uniform q        ("importance")
  D.129  adjoint correction A_{y,q} = A_y + phi d_t log q ("adjoint")
  D.133  ordered logit-normal density
  D.135  d_t log f for the logit-normal
  D.136  endpoint-aware boundary term B_phi(r)            ("endpoint" family)

Design notes that are easy to get wrong:

*   The objective is ||E h||^2, not E||h||^2, so the U statistic (D.75) is
    used and its estimate is allowed to be negative.  Never clamp, abs or
    detach it.
*   A non-uniform q is drawn from a defensive mixture with the uniform
    triangle, q >= 2 * mixture_alpha.  That bound is what keeps the
    importance weight (<= 1/alpha) and the adjoint correction finite; a bare
    logit-normal has unbounded 2/q near the corners of the triangle.
*   `diag_probability` belongs to the strong-form baselines only.  The weak
    method computes its diagonal term on a separate, always-drawn input, and
    weights it with `diag_weight`.
"""

import math

import torch

_EPS = 1e-6
_TINY = 1e-30


# ---------------------------------------------------------------------------
# time sampling
# ---------------------------------------------------------------------------


def _logit_normal_log_density(s, mean, std):
    """log f(s) for s = sigmoid(N(mean, std^2)); see D.132."""
    s = s.clamp(_EPS, 1 - _EPS)
    ell = torch.log(s) - torch.log1p(-s)
    return (
        -math.log(std)
        - 0.5 * math.log(2 * math.pi)
        - torch.log(s)
        - torch.log1p(-s)
        - (ell - mean).square() / (2 * std * std)
    )


def _logit_normal_dlog_density(s, mean, std):
    """d/ds log f(s); see D.135."""
    s = s.clamp(_EPS, 1 - _EPS)
    ell = torch.log(s) - torch.log1p(-s)
    return -1.0 / s + 1.0 / (1 - s) - (ell - mean) / (std * std * s * (1 - s))


def _sample_logit_normal(shape, mean, std, device, dtype):
    normal = torch.randn(shape, device=device, dtype=dtype)
    return torch.sigmoid(normal * std + mean).clamp(_EPS, 1 - _EPS)


def sample_uniform_triangle(batch, device, dtype=torch.float32):
    """Sorting iid uniforms gives q(r,t)=2 on 0<r<t<1 (in exact arithmetic)."""
    pair = torch.rand(batch, 2, device=device, dtype=dtype)
    r = pair.min(dim=1).values
    t = pair.max(dim=1).values
    return t[:, None, None, None], r[:, None, None, None]


def sample_time_pair(batch, device, dtype, sampler="uniform", mixture_alpha=0.2,
                     mean_t=-0.6, std_t=1.6, mean_r=-4.0, std_r=1.6):
    """Draw (t, r) on the open triangle and report q and d_t log q.

    Returns flat tensors (t, r, q, dlogq_dt) of shape [B].  For the uniform
    sampler q is identically 2 and d_t log q is 0.

    The non-uniform sampler is the defensive mixture

        q(r,t) = 2 * alpha + (1 - alpha) * [f_t(t) f_r(r) + f_t(r) f_r(t)]

    for 0 < r < t < 1, where f_t, f_r are logit-normal densities.  The second
    bracket is the density of ordering two independent draws (D.133 is the
    special case f_t = f_r).  Mixing in the uniform triangle guarantees
    q >= 2 alpha everywhere on the triangle.
    """
    if sampler == "uniform":
        pair = torch.rand(batch, 2, device=device, dtype=dtype)
        r = pair.min(dim=1).values
        t = pair.max(dim=1).values
        q = torch.full_like(t, 2.0)
        return t, r, q, torch.zeros_like(t)

    if sampler != "logitnormal":
        raise ValueError(f"Unknown weak time sampler: {sampler}")
    if not 0.0 < mixture_alpha <= 1.0:
        raise ValueError("mixture_alpha must lie in (0, 1]")

    uniform_pair = torch.rand(batch, 2, device=device, dtype=dtype)
    heavy = torch.stack(
        (
            _sample_logit_normal(batch, mean_t, std_t, device, dtype),
            _sample_logit_normal(batch, mean_r, std_r, device, dtype),
        ),
        dim=1,
    )
    take_uniform = (torch.rand(batch, device=device, dtype=dtype) < mixture_alpha)
    pair = torch.where(take_uniform[:, None], uniform_pair, heavy)
    r = pair.min(dim=1).values.clamp(_EPS, 1 - _EPS)
    t = pair.max(dim=1).values.clamp(_EPS, 1 - _EPS)

    log_ft_t = _logit_normal_log_density(t, mean_t, std_t)
    log_fr_r = _logit_normal_log_density(r, mean_r, std_r)
    log_ft_r = _logit_normal_log_density(r, mean_t, std_t)
    log_fr_t = _logit_normal_log_density(t, mean_r, std_r)
    # Both orderings map to the same (r, t): f_t(t) f_r(r) + f_t(r) f_r(t).
    ordered = torch.exp(log_ft_t + log_fr_r) + torch.exp(log_ft_r + log_fr_t)
    q = 2.0 * mixture_alpha + (1.0 - mixture_alpha) * ordered

    # d_t of the ordered density; f'(s) = f(s) * dlogf(s), zeroed on underflow.
    def _deriv(log_f, s, mean, std):
        f = torch.exp(log_f)
        d = _logit_normal_dlog_density(s, mean, std)
        return torch.where(f > _TINY, f * d, torch.zeros_like(f))

    d_ordered = (
        _deriv(log_ft_t, t, mean_t, std_t) * torch.exp(log_fr_r)
        + torch.exp(log_ft_r) * _deriv(log_fr_t, t, mean_r, std_r)
    )
    dlogq_dt = (1.0 - mixture_alpha) * d_ordered / q
    return t, r, q, dlogq_dt


# ---------------------------------------------------------------------------
# test functions
# ---------------------------------------------------------------------------


def fourier_coefficients(z, velocity, t, r, Wz, br, ct, phase,
                         family="vanishing", dlogq_dt=None):
    """Wz:[M,d], all other feature parameters:[M], shared across samples.

    family="vanishing": phi = (1-t) cos xi                      (D.43)
    family="endpoint":  phi = cos xi, which does not vanish at t=1 and so
                        keeps the upper boundary term of D.136 alive.

    `dlogq_dt` (flat [B]) adds the adjoint correction phi * d_t log q (D.129).
    """
    z = z.flatten(1)
    velocity = velocity.flatten(1)
    t, r = t.reshape(-1, 1), r.reshape(-1, 1)
    xi = z @ Wz.T + r * br + t * ct + phase
    cos_xi, sin_xi = xi.cos(), xi.sin()
    projected = ct + velocity @ Wz.T
    if family == "vanishing":
        phi = (1 - t) * cos_xi
        A = -cos_xi - (1 - t) * sin_xi * projected
    elif family == "endpoint":
        phi = cos_xi
        A = -sin_xi * projected
    else:
        raise ValueError(f"Unknown test-function family: {family}")
    if dlogq_dt is not None:
        A = A + phi * dlogq_dt.reshape(-1, 1)
    return A, phi


# ---------------------------------------------------------------------------
# U statistic
# ---------------------------------------------------------------------------


def u_statistic_terms(coefficients, vectors):
    """Unbiased estimate of ||E h||^2 / d for h_i = sum_p C^(p)_i (x) V^(p)_i.

    `coefficients` is a list of [B, M] tensors, `vectors` a list of matching
    [B, d] tensors.  With one pair this is the plain D.99 form; extra pairs let
    the endpoint boundary term join the same estimator.  Cost is O(P^2 B (M+d)),
    never materialising B x M x d.
    """
    if len(coefficients) != len(vectors) or not coefficients:
        raise ValueError("coefficients and vectors must be non-empty and equal length")
    batch = vectors[0].shape[0]
    dim = vectors[0].shape[1]
    features = coefficients[0].shape[1]
    if batch < 2:
        raise ValueError("U statistic requires at least two independent samples")

    total = sum(C.T @ V for C, V in zip(coefficients, vectors))
    diagonal = vectors[0].new_zeros(())
    for p, (Cp, Vp) in enumerate(zip(coefficients, vectors)):
        for q, (Cq, Vq) in enumerate(zip(coefficients, vectors)):
            gram = (Vp * Vq).sum(dim=1)              # [B]
            weight = (Cp * Cq).sum(dim=1)            # [B]
            diagonal = diagonal + (gram * weight).sum()
    # Negative estimates are valid. Do not clamp, abs(), or detach diagonal.
    return (total.square().sum() - diagonal) / (batch * (batch - 1) * features * dim)


def u_statistic(F, velocity, A, phi):
    """Backwards-compatible two-term form: h_i = A_i F_i + phi_i y_i (D.99)."""
    return u_statistic_terms([A, phi], [F.flatten(1), velocity.flatten(1)])


# ---------------------------------------------------------------------------
# weak loss
# ---------------------------------------------------------------------------


def random_fourier_loss(z, velocity, F, t, r, features=64,
                        sigma_z=1.0, sigma_r=1.0, sigma_t=1.0,
                        fp64=False, family="vanishing", weights=None,
                        dlogq_dt=None, boundary=None):
    """One weak-loss evaluation for an already-computed network output.

    `weights` (flat [B]) carries the importance correction 2/q (D.123); it
    multiplies the interior contribution only.  `boundary`, when given, is a
    pair (noise, F_boundary, r_boundary) whose contribution 2 cos(xi) F is
    subtracted inside the same U statistic (D.136 - D.138).
    """
    dtype = torch.float64 if fp64 else torch.float32
    z, velocity, F = z.to(dtype), velocity.to(dtype), F.to(dtype)
    t, r = t.to(dtype), r.to(dtype)
    dim = z[0].numel()
    if dlogq_dt is not None:
        dlogq_dt = dlogq_dt.to(dtype)
    with torch.no_grad():
        # Per-update resampling; independent of the batch data; shared in batch.
        Wz = torch.randn(features, dim, device=z.device, dtype=dtype) * (sigma_z / math.sqrt(dim))
        br = torch.randn(features, device=z.device, dtype=dtype) * sigma_r
        ct = torch.randn(features, device=z.device, dtype=dtype) * sigma_t
        phase = torch.rand(features, device=z.device, dtype=dtype) * (2 * math.pi)
        A, phi = fourier_coefficients(z, velocity, t, r, Wz, br, ct, phase,
                                      family=family, dlogq_dt=dlogq_dt)
        if weights is not None:
            w = weights.to(dtype).reshape(-1, 1)
            A, phi = A * w, phi * w

    coefficients = [A, phi]
    vectors = [F.flatten(1), velocity.flatten(1)]

    if boundary is not None:
        noise, F_boundary, r_boundary = boundary
        noise = noise.to(dtype)
        F_boundary = F_boundary.to(dtype)
        r_boundary = r_boundary.to(dtype).reshape(-1, 1)
        with torch.no_grad():
            xi_b = noise.flatten(1) @ Wz.T + r_boundary * br + ct + phase
            # 2 * E_{r~U(0,1)}[B_phi(r)] matches the 2 * int_0^1 . dr weight
            # that the interior term carries through q0 = 2.
            K = 2.0 * xi_b.cos()
        # h = boundary - interior, so the interior pair flips sign.
        coefficients = [-A, -phi, K]
        vectors = [F.flatten(1), velocity.flatten(1), F_boundary.flatten(1)]

    return u_statistic_terms(coefficients, vectors)


def _weak_settings(args):
    if getattr(args, "diag_probability", None) is not None:
        raise ValueError(
            "diag_probability applies to method=mf_control/imf_diag only. "
            "The weak method always evaluates its diagonal term; use diag_weight."
        )
    family = str(getattr(args, "weak_test_family", "vanishing"))
    sampler = str(getattr(args, "weak_time_sampler", "uniform"))
    correction = str(getattr(args, "weak_time_correction", "importance"))
    diag_time = str(getattr(args, "weak_diag_time", "uniform"))
    if family not in {"vanishing", "endpoint"}:
        raise ValueError(f"Unknown test-function family: {family}")
    if sampler not in {"uniform", "logitnormal"}:
        raise ValueError(f"Unknown weak time sampler: {sampler}")
    if correction not in {"importance", "adjoint"}:
        raise ValueError(f"Unknown weak time correction: {correction}")
    if diag_time not in {"uniform", "logitnormal"}:
        raise ValueError(f"Unknown weak diagonal time sampler: {diag_time}")
    if family == "endpoint" and sampler != "uniform" and correction == "adjoint":
        # D.128 weights the upper boundary by q(r,1), so the endpoint term no
        # longer shares the uniform-triangle weight 2. Importance correction
        # (D.121) keeps the uniform objective and is the supported pairing.
        raise ValueError(
            "weak_test_family=endpoint with weak_time_correction=adjoint is not "
            "supported; use weak_time_correction=importance."
        )
    return dict(
        features=int(getattr(args, "weak_features", 64)),
        weak_weight=float(getattr(args, "weak_weight", 1.0)),
        diag_weight=float(getattr(args, "diag_weight", 1.0)),
        sigma_z=float(getattr(args, "weak_sigma_z", 1.0)),
        sigma_r=float(getattr(args, "weak_sigma_r", 1.0)),
        sigma_t=float(getattr(args, "weak_sigma_t", 1.0)),
        fp64=bool(getattr(args, "weak_fp64", False)),
        family=family,
        sampler=sampler,
        correction=correction,
        mixture_alpha=float(getattr(args, "weak_mixture_alpha", 0.2)),
        mean_t=float(getattr(args, "weak_P_mean_t", -0.6)),
        std_t=float(getattr(args, "weak_P_std_t", 1.6)),
        mean_r=float(getattr(args, "weak_P_mean_r", -4.0)),
        std_r=float(getattr(args, "weak_P_std_r", 1.6)),
        diag_time=diag_time,
    )


def weak_terms(net, x, args):
    """Return (diag_loss, weak_loss, logs) without summing them.

    Keeping the two terms separate lets the trainer backward them one at a
    time, so the peak activation memory is the larger graph rather than the
    sum of both.  The gradients are identical either way.
    """
    cfg = _weak_settings(args)
    x = x.float()
    batch, device, dtype = len(x), x.device, x.dtype

    t_flat, r_flat, q, dlogq_dt = sample_time_pair(
        batch, device, dtype, sampler=cfg["sampler"],
        mixture_alpha=cfg["mixture_alpha"], mean_t=cfg["mean_t"],
        std_t=cfg["std_t"], mean_r=cfg["mean_r"], std_r=cfg["std_r"],
    )
    t = t_flat.view(-1, 1, 1, 1)
    r = r_flat.view(-1, 1, 1, 1)

    weights, adjoint = None, None
    if cfg["sampler"] != "uniform":
        if cfg["correction"] == "importance":
            weights = 2.0 / q
        elif cfg["correction"] == "adjoint":
            adjoint = dlogq_dt
        else:
            raise ValueError(f"Unknown weak time correction: {cfg['correction']}")

    velocity = torch.randn_like(x) - x
    z = x + t * velocity
    u = net(z, (t.flatten(), (t - r).flatten()), None)
    F = (t - r) * u

    boundary = None
    if cfg["family"] == "endpoint":
        # D.136: B_phi(r) = E_eps[phi(eps, r, 1) F_theta(eps, r, 1)], with
        # F_theta(eps, r, 1) = (1 - r) u_theta(eps, r, 1).  r is drawn
        # uniformly so no importance weight is needed for this term.
        r_b = torch.rand(batch, device=device, dtype=dtype).view(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        ones = torch.ones_like(r_b)
        u_b = net(noise, (ones.flatten(), (ones - r_b).flatten()), None)
        boundary = (noise, (1 - r_b) * u_b, r_b.flatten())

    weak = random_fourier_loss(
        z, velocity, F, t, r, cfg["features"], cfg["sigma_z"], cfg["sigma_r"],
        cfg["sigma_t"], cfg["fp64"], family=cfg["family"], weights=weights,
        dlogq_dt=adjoint, boundary=boundary,
    )

    # Separate diagonal input; the diagonal term is always evaluated.
    if cfg["diag_time"] == "uniform":
        td = torch.rand_like(t)
    elif cfg["diag_time"] == "logitnormal":
        td = _sample_logit_normal(batch, cfg["mean_t"], cfg["std_t"], device, dtype).view(-1, 1, 1, 1)
    else:
        raise ValueError(f"Unknown weak diagonal time sampler: {cfg['diag_time']}")
    vd = torch.randn_like(x) - x
    zd = x + td * vd
    ud = net(zd, (td.flatten(), torch.zeros_like(td).flatten()), None)
    # Normalized diagonal loss = Appendix D.115 / d.
    diag = (ud - vd).square().mean()

    logs = {
        "diag_mse": diag.detach(),
        "weak_u": weak.detach(),
        "weighted_weak": (cfg["weak_weight"] * weak).detach(),
        "weight_max": (weights.max() if weights is not None else torch.zeros((), device=device)),
        "q_min": q.min(),
    }
    return cfg["diag_weight"] * diag, cfg["weak_weight"] * weak, logs


def experimental_loss(net, x, args):
    """Controlled MF / diagonal iMF / weak losses, with ordinary unweighted MSE."""
    x = x.float()
    if args.method == "weak":
        diag, weak, logs = weak_terms(net, x, args)
        total = diag + weak
        return total, {name: value.detach() for name, value in logs.items()}

    t, r = sample_uniform_triangle(len(x), x.device)
    # Same controlled sampler for MF and diagonal iMF, plus diagonal mass.
    mask = torch.rand_like(t) < args.diag_probability
    r = torch.where(mask, t, r)
    velocity = torch.randn_like(x) - x
    z = x + t * velocity

    def u_func(zz, tt, rr):
        return net(zz, (tt.flatten(), (tt - rr).flatten()), None)

    u = u_func(z, t, r)
    with torch.no_grad():
        direction = velocity if args.method == "mf_control" else u_func(z, t, t)
        _, dudt = torch.func.jvp(
            u_func, (z, t, r),
            (direction, torch.ones_like(t), torch.zeros_like(r)),
        )
    target = (velocity - (t - r) * dudt).detach()
    total = (u - target).square().mean()
    return total, {"strong_mse": total.detach()}
