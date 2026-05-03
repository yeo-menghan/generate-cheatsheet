# Lecture 9: Generative Models & VAEs

## 1. Discriminative vs Generative Models

| | Discriminative | Generative |
|---|---|---|
| **Goal** | Learn $f^*(\mathbf{x})$ (decision boundary) | Learn $p^*(\mathbf{x})$ (data distribution) |
| **Output** | Labels / predictions | New samples $\tilde{\mathbf{x}} \sim p^*$ |

> **Intuition:** Discriminative models learn to *classify* — generative models 
> learn to *create*. A discriminative model for faces learns "is this a face?"; 
> a generative model learns "what does a face look like?"


## 2. Density Estimation vs Generative Models

- **Density Estimation**: Find $\hat{p} \approx p^*$ — may not be easily samplable.
- **Generative Models**: Sample $\tilde{\mathbf{x}} \sim p^*$ approximately.

> The distinction matters: you can estimate a distribution without being able 
> to efficiently draw samples from it.

## 3. Gaussian Mixture Models (GMM)

$$p_{\boldsymbol{\theta}}(\mathbf{x}) = \sum_j \alpha_j \, p_{\boldsymbol{\mu}_j, \Sigma_j}(\mathbf{x}), \quad \alpha_j \geq 0, \quad \sum_j \alpha_j = 1$$

**Sampling**: (1) Sample index $i$ from $\{\alpha_j\}$; (2) Sample from 
$p_{\boldsymbol{\mu}_i, \Sigma_i}$.

**Limitations**: Poor for high-dimensional data; $K$ is hard to tune; cannot 
leverage domain knowledge (e.g., CNNs).

> **Intuition:** GMM assumes data comes from $K$ Gaussian clusters. Simple and 
> interpretable, but breaks down for complex high-dimensional data like images.

## 4. Maximum Likelihood Estimation (MLE)

Given dataset $\mathcal{D} = \{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(N)}\}$ 
with $\mathbf{x}^{(i)} \sim p^*$, find parameters that make the observed data 
**most probable**:

$$\hat{\boldsymbol{\theta}} = \arg\max_{\boldsymbol{\theta}} \sum_i \log p_{\boldsymbol{\theta}}(\mathbf{x}^{(i)})$$

> We use log-likelihood instead of likelihood for numerical stability — 
> products of small probabilities underflow; sums of logs don't.

**Gaussian MLE** ($\boldsymbol{\theta} = (\mu, \sigma)$):

$$\hat{\mu} = \frac{1}{N}\sum_{i=1}^N x^{(i)}, \qquad \hat{\sigma}^2 = \frac{1}{N}\sum_{i=1}^N (x^{(i)} - \hat{\mu})^2$$

## 5. Latent Generative Models

Real-world data is complex — rather than modeling $p(\mathbf{x})$ directly, 
introduce a **latent variable** $\mathbf{z}$ that explains the underlying 
structure:

$$p_{\boldsymbol{\theta}}(\mathbf{x}) = \int p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z})\, p_{\boldsymbol{\theta}}(\mathbf{z})\, d\mathbf{z}$$

- $\mathbf{z}$: latent variable — compact representation of hidden structure
- $p_{\boldsymbol{\theta}}(\mathbf{z})$: prior — what we assume about latent space 
  before seeing data (typically $\mathcal{N}(\mathbf{0}, I)$)
- $p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z})$: decoder — given latent code, 
  generate data

> **Intuition:** For face generation, $\mathbf{z}$ might encode "smile intensity", 
> "hair color", "age" — the decoder then renders a face from these attributes.

## 6. Variational Autoencoder (VAE)

### AE vs VAE — The Key Difference

| Component | AE | VAE |
|---|---|---|
| **Encoder** | $\mathbf{z} = f_{\boldsymbol{\phi}}(\mathbf{x})$ — single point | $\mathbf{z} \sim q_{\boldsymbol{\phi}}(\mathbf{z}\|\mathbf{x})$ — distribution |
| **Decoder** | $\mathbf{x}' = g_{\boldsymbol{\theta}}(\mathbf{z})$ | $\mathbf{x}' \sim p_{\boldsymbol{\theta}}(\mathbf{x}\|\mathbf{z})$ |

**Why distributions instead of points?**

A regular AE maps each input to a **single point** in latent space — no 
guarantee of structure or continuity. Points between learned latent vectors 
may decode to **garbage** because nothing was trained there.

VAE forces the latent space to be a **continuous, structured Gaussian** — 
you can sample *anywhere* in latent space and decode something meaningful.

**The posterior problem:**

The true posterior $p_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})$ is 
**intractable** (requires integrating over all $\mathbf{z}$). Solution: 
approximate it with a learned distribution:
$$q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}) \approx p_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})$$


## 7. KL Divergence

Measures how different two distributions are:

$$D_{KL}(p \| q) = \int p(\mathbf{x}) \log \frac{p(\mathbf{x})}{q(\mathbf{x})}\, d\mathbf{x} = \mathbb{E}_{p(\mathbf{x})}\!\left[\log \frac{p(\mathbf{x})}{q(\mathbf{x})}\right]$$

**Properties:**
- $D_{KL}(p\|q) \geq 0$ — always non-negative
- $D_{KL}(p\|q) = 0$ iff $p = q$ — zero only when distributions are identical
- **Not symmetric**: $D_{KL}(p\|q) \neq D_{KL}(q\|p)$

**Gaussian KL** ($p_{\boldsymbol{\theta}_1} = \mathcal{N}(\mu_1, \sigma_1^2)$, 
$p_{\boldsymbol{\theta}_2} = \mathcal{N}(\mu_2, \sigma_2^2)$):

$$D_{KL}(p_{\boldsymbol{\theta}_1} \| p_{\boldsymbol{\theta}_2}) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2}{2\sigma_2^2} - \frac{1}{2} + \frac{(\mu_1 - \mu_2)^2}{2\sigma_2^2}$$


## 8. Monte-Carlo Estimation

When an expectation is intractable analytically, approximate it by averaging 
over samples:

$$\mathbb{E}[f(\mathbf{x})] \approx \frac{1}{N}\sum_{i=1}^N f(\mathbf{x}^{(i)}), \quad \mathbf{x}^{(i)} \sim p \text{ i.i.d.}$$

> Used in VAE training to approximate the ELBO expectation over $\mathbf{z}$.

## 9. Evidence Lower Bound (ELBO)

Since $\log p_{\boldsymbol{\theta}}(\mathbf{x})$ is intractable, we instead 
maximize a **tractable lower bound**:

$$\log p_{\boldsymbol{\theta}}(\mathbf{x}) = \underbrace{\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})}\!\left[\log \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})}\right]}_{L(\mathbf{x};\,\boldsymbol{\theta},\boldsymbol{\phi})\ \text{(ELBO)}} + \underbrace{D_{KL}\!\left(q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}) \| p_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x})\right)}_{\geq\, 0}$$

Since KL $\geq 0$: $\;\log p_{\boldsymbol{\theta}}(\mathbf{x}) \geq L(\mathbf{x};\boldsymbol{\theta},\boldsymbol{\phi})$

**Training objective**: $\max_{\boldsymbol{\theta},\boldsymbol{\phi}}\ L(\mathbf{x};\boldsymbol{\theta},\boldsymbol{\phi})$

> **Why ELBO works:** Maximizing the ELBO simultaneously maximizes the 
> log-likelihood AND minimizes the gap between $q_{\boldsymbol{\phi}}$ and 
> the true posterior — killing two birds with one stone.

**Multi-sample ELBO**:
$$L(\mathcal{D};\boldsymbol{\theta},\boldsymbol{\phi}) = \frac{1}{N}\sum_{i=1}^N \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}^{(i)})}\!\left[\log \frac{p_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}^{(i)})}\right]$$

## 10. Reparameterization Trick

**The problem:** Sampling $\mathbf{z} \sim q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})$ 
directly is **not differentiable** — the sampling operation is a stochastic node 
in the computational graph. Backprop cannot flow gradients through it since there 
is no deterministic relationship between $\boldsymbol{\phi}$ and $\mathbf{z}$.

**The fix:** Move the randomness outside the graph by rewriting as a 
**deterministic transformation**:

$$\mathbf{z} = g(\mathbf{u}, \boldsymbol{\phi}, \mathbf{x}), \quad \mathbf{u} \sim p_0(\mathbf{u}) \text{ (fixed, independent of } \boldsymbol{\phi})$$

**Gaussian example**:
$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \circ \mathbf{u}, \quad \mathbf{u} \sim \mathcal{N}(\mathbf{0}, I)$$

Now $\mathbf{u}$ is sampled from a **fixed distribution independent of 
$\boldsymbol{\phi}$** — the path from $\boldsymbol{\phi}$ to $\mathbf{z}$ 
is deterministic and differentiable:
$$\nabla_{\boldsymbol{\phi}}\mathbf{z} = \nabla_{\boldsymbol{\phi}}(\boldsymbol{\mu} + \boldsymbol{\sigma} \circ \mathbf{u})$$

**Enables gradient flow:**
$$\nabla_{\boldsymbol{\phi}}\,\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})}[f(\mathbf{z})] = \mathbb{E}_{p_0(\mathbf{u})}\!\left[\nabla_{\boldsymbol{\phi}} f(g(\mathbf{u}, \boldsymbol{\phi}, \mathbf{x}))\right]$$

> **Analogy:** Instead of "sample a random number that depends on $\boldsymbol{\phi}$" 
> → "sample a standard random number, then transform it using $\boldsymbol{\phi}$." 
> Same result, but now $\boldsymbol{\phi}$ sits on a differentiable path.

## 11. Neural Network Parameterization

### Encoder — $q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})$ (Factorized Gaussian)

Outputs the **mean and log-variance** of the latent distribution:
$$(\boldsymbol{\mu},\, \log\boldsymbol{\sigma}) = \text{EncodingNN}(\mathbf{x};\boldsymbol{\phi})$$
$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \circ \mathbf{u}, \quad \mathbf{u} \sim \mathcal{N}(\mathbf{0}, I)$$

> We output $\log\boldsymbol{\sigma}$ instead of $\boldsymbol{\sigma}$ directly 
> to keep values unconstrained — avoids forcing the network to output 
> strictly positive numbers.

### Decoder — $p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z})$ (Factorized Bernoulli for binary data)

$$\mathbf{s} = \text{DecodingNN}(\mathbf{z};\boldsymbol{\theta}), \quad \mathbf{s} \in [0,1]^d$$
$$\log p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z}) = \sum_j \left[x_j \log s_j + (1-x_j)\log(1-s_j)\right] = -\text{BCE}(\mathbf{x}, \mathbf{s})$$

### Prior — $p_{\boldsymbol{\theta}}(\mathbf{z})$

$$p_{\boldsymbol{\theta}}(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I) \quad \Rightarrow \quad \log p_{\boldsymbol{\theta}}(\mathbf{z}) = -\frac{1}{2}\sum_j \left[z_j^2 + \log(2\pi)\right]$$

## 12. VAE Loss (Basic Form)

Let $\mathbf{y}_1 = \boldsymbol{\mu}$, $\mathbf{y}_2 = \log\boldsymbol{\sigma}$ 
be encoder outputs.

$$\boxed{-L(\mathbf{x};\boldsymbol{\theta},\boldsymbol{\phi}) = \underbrace{\text{BCE}(\mathbf{x}, \mathbf{s})}_{\text{Reconstruction Loss}} + \underbrace{\frac{1}{2}\|\mathbf{y}_1\|^2 + \frac{1}{2}\|e^{\mathbf{y}_2}\|^2 - \sum_j y_{2,j}}_{\text{KL Divergence Loss}}}$$

**Equivalently**:
$$-L = -\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x})}\!\left[\log p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z})\right] + D_{KL}\!\left(q_{\boldsymbol{\phi}}(\mathbf{z}|\mathbf{x}) \| p_{\boldsymbol{\theta}}(\mathbf{z})\right)$$

### The Two Terms and Their Tension

| Term | Role | Too much weight → |
|---|---|---|
| **Reconstruction (BCE)** | Faithfully reconstruct input | Encoder ignores prior → latent space collapses to points → poor generation |
| **KL Divergence** | Force latent space toward $\mathcal{N}(\mathbf{0}, I)$ | Encoder ignores input → all inputs map to same distribution → poor reconstruction |

> **The tension:** Reconstruction wants the encoder to be **input-specific** 
> (different $\mathbf{z}$ for different $\mathbf{x}$), while KL wants all 
> encodings to look like the **same standard Gaussian**. The balance produces 
> a structured yet expressive latent space.

## 13. Key Takeaways

- **AEs** produce a discontinuous latent space → poor generation.
- **VAEs** enforce a structured (Gaussian) latent space → smooth, continuous generation.
- The **ELBO** is a tractable lower bound on the log-likelihood, avoiding 
  intractable posterior computation.
- The **reparameterization trick** moves randomness outside the computational 
  graph → enables gradient flow through stochastic sampling.
- The VAE loss balances **reconstruction quality** (BCE) and **latent 
  regularization** (KL divergence toward $\mathcal{N}(\mathbf{0}, I)$).