import numpy as np

def reparameterize(mu, logvar, eps):
    """z = mu + exp(0.5 * logvar) * eps, all shapes (batch, latent_dim)."""
    sigma = np.exp(0.5 * logvar)
    z = mu + sigma * eps
    return z

def kl_to_standard_normal(mu, logvar):
    """KL(N(mu, sigma^2) || N(0, I)) summed over latent dims, averaged over the batch. Returns float."""
    kl_elements = 0.5 * (mu**2 + np.exp(logvar) - 1 - logvar)
    kl_per_sample = np.sum(kl_elements, axis=1)
    kl = np.mean(kl_per_sample)
    return float(kl)