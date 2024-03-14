from scipy import integrate
import torch
import numpy as np

def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, eps, class_labels=None):
    with torch.enable_grad():
      x.requires_grad_(True)
      fn_eps = torch.sum(fn(x, t, class_labels) * eps)
      grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    x.requires_grad_(False)
    return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

  return div_fn

def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))

def get_likelihood_fn(MODEL, hutchinson_type='Rademacher',
                      rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):
  """Create a function to compute the unbiased log-likelihood estimate of a given data point.

  Args:
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    inverse_scaler: The inverse data normalizer.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
      See documentation for `scipy.integrate.solve_ivp`.
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

  Returns:
    A function that a batch of data points and returns the log-likelihoods in bits/dim,
      the latent code, and the number of function evaluations cost by computation.
  """

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)


  def drift_fn(model, x, t, class_labels=None):
    """The drift function of the reverse-time SDE."""
    sigma = t
    D_x = model(x, sigma, class_labels)
    S_x = (D_x - x)/(sigma**2)
    return S_x

  def div_fn(model, x, t, noise, class_labels=None):
    return get_div_fn(lambda xx, tt, class_labels: drift_fn(model, xx, tt, class_labels))(x, t, noise, class_labels)

  def likelihood_fn(model, data, class_labels=None):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Args:
      model: A score model.
      data: A PyTorch tensor.

    Returns:
      bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
      z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """
    with torch.no_grad():
      shape = data.shape
      if hutchinson_type == 'Gaussian':
        epsilon = torch.randn_like(data)
      elif hutchinson_type == 'Rademacher':
        epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
      else:
        raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

      def ode_func(t, x):
        sample = from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
        vec_t = torch.ones(sample.shape[0], device=sample.device) * t
        drift = to_flattened_numpy(drift_fn(model, sample, vec_t, class_labels))
        logp_grad = to_flattened_numpy(div_fn(model, sample, vec_t, epsilon, class_labels))
        return np.concatenate([drift, logp_grad], axis=0)

      init = np.concatenate([to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
      solution = integrate.solve_ivp(ode_func, (eps, MODEL.sigma_max), init, rtol=rtol, atol=atol, method=method)
      #nfe = solution.nfev
      zp = solution.y[:, -1]
      z = from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
      delta_logp = from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
            
      prior_logp = prior_logp(z)

      post_logp = prior_logp + delta_logp
      #bpd = -(prior_logp + delta_logp) / np.log(2)
      #N = np.prod(shape[1:])
      #bpd = bpd / N
      # A hack to convert log-likelihoods to bits/dim
      #offset = 7. - inverse_scaler(-1.)
      #bpd = bpd + offset
      return zp, post_logp #bpd, z, nfe

  return likelihood_fn