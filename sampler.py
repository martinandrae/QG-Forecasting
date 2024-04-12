import torch
import numpy as np

def edm_sampler(
    net, latents, class_labels=None, time_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    time_labels = time_labels.to(torch.float32).to(latents.device)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents * t_steps[0]
    #xs = [x_next]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels, time_labels)#.to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels, time_labels)#.to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
       # xs.append(x_next)

    return x_next #torch.stack(xs) #xs[::-1])


def heun_sampler(
    net, latents, class_labels=None, time_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    time_labels = time_labels.to(torch.float32).to(latents.device)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Euler step.
        denoised = net(x_cur, t_cur, class_labels, time_labels)
        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels, time_labels)
            d_prime = (x_next - denoised) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
        
    return x_next


def complete_edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents * t_steps[0]
    xs = [x_next]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels)#.to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels)#.to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        xs.append(x_next)

    return x_next, torch.stack(xs) #xs[::-1])

def drift_fn(model, x, t, class_labels=None, time_labels=None):
    """The drift function of the reverse-time SDE."""
    sigma = t
    D_x = model(x, sigma, class_labels, time_labels)
    S_x = (D_x - x)/(sigma**2)
    return S_x


def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, eps, class_labels=None):
    with torch.enable_grad():
      x.requires_grad_(True)
      fn_eps = torch.sum(fn(x, t, class_labels) * eps, axis=(1,2,3))
      grad_fn_eps = torch.zeros_like(eps)

      for i in range(eps.size(0)):
          dl = torch.autograd.grad(fn_eps[i], x, create_graph=True, allow_unused=True)[0]
          grad_fn_eps[i] = dl

    #x.requires_grad_(False)
    res = torch.sum(grad_fn_eps * eps, dim=(1,2,3)).mean(dim=0)
    return res

  return div_fn


def div_fn(model, x, t, noise, class_labels=None, time_labels=None):
    return get_div_fn(lambda xx, tt, class_labels, time_labels: drift_fn(model, xx, tt, class_labels, time_labels))(x, t, noise, class_labels, time_labels)


def likelihood_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    eps_multiplier=1
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents * t_steps[0]

    eps_shape = x_next.shape
    eps_shape = (eps_shape[0] * eps_multiplier,) + eps_shape[1:]  # Double the number of channels
    epsilon = torch.randn(eps_shape).to(latents.device)
    
    div_next = div_fn(net, x_next, t_steps[0], epsilon, class_labels)

    N = np.prod(latents.shape[1:])
    prior_logp = -N / 2. * np.log(2 * np.pi * sigma_max ** 2) - torch.sum(latents ** 2, dim=(1, 2, 3)) / (2 * sigma_max ** 2)
    prior_logp = prior_logp[0]
    post_logp = prior_logp

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # SAMPLE
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels)#.to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        # LIKELIHOOD
        div_cur = div_next
        
        if i < num_steps - 1:
            epsilon = torch.randn(eps_shape).to(latents.device)
            div_next = div_fn(net, x_next, t_next, epsilon, class_labels)
            delta_logp = (div_cur + div_next) * (t_next - t_cur) / 2
            
            post_logp += delta_logp

    return x_next, post_logp


def likelihood_estimator(
    net, latents, samples, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    eps_multiplier=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = samples[0]
    
    eps_shape = x_next.shape
    eps_shape = (eps_shape[0] * eps_multiplier,) + eps_shape[1:]  # Double the number of channels
    epsilon = torch.randn(eps_shape).to(latents.device)
    #print(epsilon.shape)
    div_next = div_fn(net, x_next, t_steps[0], epsilon, class_labels)

    N = np.prod(latents.shape[1:])
    prior_logp = -N / 2. * np.log(2 * np.pi * sigma_max ** 2) - torch.sum(latents ** 2, dim=(1, 2, 3)) / (2 * sigma_max ** 2)
    prior_logp = prior_logp[0]
    #prior_logp = torch.tensor(0.0).to(device)
    post_logp = prior_logp# torch.tensor(0.0).to(device) # torch.zeros_like(prior_logp) #prior_logp

    #logps = [prior_logp]
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_next = samples[i + 1]

        # Likelihood
        div_cur = div_next
        
        if i < num_steps - 1:
            epsilon = torch.randn(eps_shape).to(latents.device)
            div_next = div_fn(net, x_next, t_next, epsilon, class_labels)
            delta_logp = (div_cur + div_next) * (t_next - t_cur) / 2
            
            post_logp += delta_logp
            #logps.append(delta_logp)

    return post_logp #, torch.stack(logps)
