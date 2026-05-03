# Lecture 10: GANs & Neural Style Transfer

## 1. Generative Adversarial Networks (GANs)

### Setup

| Component | Definition |
|-----------|-----------|
| Latent variable | $\mathbf{z} \sim p_0 = \mathcal{N}(0, I_n)$ |
| Training data | $\{\mathbf{x}^{(i)} \in \mathbb{R}^d : i=1,\ldots,N,\ \mathbf{x}^{(i)} \sim p^*\}$ |
| Generator | $G_{\boldsymbol{\theta}} : \mathbb{R}^n \to \mathbb{R}^d$ — maps noise to fake samples |
| Discriminator | $D_{\boldsymbol{\phi}} : \mathbb{R}^d \to [0,1]$ — predicted probability of $\mathbf{x}$ being **real** |

### Objective (Min-Max / Saddle-Point)

$$\min_{\boldsymbol{\theta}} \max_{\boldsymbol{\phi}}\ L(\boldsymbol{\theta}, \boldsymbol{\phi}) = \mathbb{E}_{\mathbf{x}\sim p^*}[\log D_{\boldsymbol{\phi}}(\mathbf{x})] + \mathbb{E}_{\mathbf{z}\sim p_0}[\log(1 - D_{\boldsymbol{\phi}}(G_{\boldsymbol{\theta}}(\mathbf{z})))]$$

- **Discriminator** maximises $L$ — learns to tell real from fake
- **Generator** minimises $L$ — learns to fool the discriminator

### Nash Equilibrium

- $p_{G_\theta} = p^*$ — generator perfectly matches true distribution
- $D_{\boldsymbol{\phi}}(\mathbf{x}) = \tfrac{1}{2}$ for all $\mathbf{x}$ — discriminator is maximally confused

> **Why $D = \frac{1}{2}$ means the generator wins:** The discriminator can do 
> no better than a random $50/50$ guess — it **cannot distinguish real from fake** 
> — implying $p_G = p^*$. The generator has perfectly replicated the true distribution.

### Basic Training Algorithm

```
Repeat:
  For k steps:
    Sample m noise {z^(i)} ~ p0, real {x^(i)} from D
    Update discriminator (ascent):
      φ ← φ + ε₁ ∇_φ (1/m) Σ [log D_φ(x^(i)) + log(1 − D_φ(G_θ(z^(i))))]
  
  Sample m noise {z^(i)} ~ p0
  Update generator (descent):
    θ ← θ − ε₂ ∇_θ (1/m) Σ log(1 − D_φ(G_θ(z^(i))))
```

## 2. Theoretical Connection: JS Divergence

### Optimal Discriminator Derivation

For fixed $G$, the discriminator maximizes:
$$L(\boldsymbol{\phi}) = \mathbb{E}_{\mathbf{x}\sim p^*}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z}\sim p_0}[\log(1 - D(G(\mathbf{z})))]$$

Rewrite as an integral over $\mathbf{x}$:
$$L = \int \left[p^*(\mathbf{x})\log D(\mathbf{x}) + p_G(\mathbf{x})\log(1 - D(\mathbf{x}))\right] d\mathbf{x}$$

For each $\mathbf{x}$, maximize $f(D) = a\log D + b\log(1-D)$ where 
$a = p^*(\mathbf{x})$, $b = p_G(\mathbf{x})$:
$$\frac{df}{dD} = \frac{a}{D} - \frac{b}{1-D} = 0$$
$$\Rightarrow a(1-D) = bD \Rightarrow \boxed{D^*(\mathbf{x}) = \frac{p^*(\mathbf{x})}{p^*(\mathbf{x}) + p_G(\mathbf{x})}}$$

At equilibrium $p_G = p^*$:
$$D^*(\mathbf{x}) = \frac{p^*(\mathbf{x})}{p^*(\mathbf{x}) + p^*(\mathbf{x})} = \frac{1}{2} \quad \checkmark$$

**GAN training ≡ minimising Jensen-Shannon divergence:**
$$\min_G D_{JS}(p_G \| p^*) - 2\log 2$$

**Jensen-Shannon Divergence:**
$$D_{JS}(p \| q) = \frac{1}{2}D_{KL}\!\left(p\,\Big\|\,\frac{p+q}{2}\right) + \frac{1}{2}D_{KL}\!\left(q\,\Big\|\,\frac{p+q}{2}\right)$$

- Non-negative, symmetric, and $D_{JS}=0 \iff p=q$
- $\sqrt{D_{JS}}$ is a metric on the space of distributions

## 3. Training Problems & Solutions

### Problem I: Vanishing Gradients

$$\nabla_{\boldsymbol{\theta}} L \propto \mathbb{E}_{\mathbf{z}\sim p_0} D_{\boldsymbol{\phi}}(G_{\boldsymbol{\theta}}(\mathbf{z})) \approx 0 \quad \text{when } D_{\boldsymbol{\phi}} \approx 0$$

Early training: bad generator → $D(G(\mathbf{z})) \approx 0$ → 
$\log(1 - D(G(\mathbf{z}))) \approx \log(1) = 0$ → **gradient vanishes**.

**Fix:** Replace $\log(1 - D(G(\mathbf{z})))$ with $-\log D(G(\mathbf{z}))$:

| Loss | Gradient near $D \approx 0$ | Gradient near $D \approx 1$ |
|---|---|---|
| $\log(1-D)$ | $\approx 0$ — **vanishes** | Large — but generator already good |
| $-\log D$ | Large — **strong signal** | $\approx 0$ — but generator already good |

$-\log D$ provides stronger gradients exactly when the generator needs them most.

**Why we still keep $\log(1-D)$ in the theory:**
- $\log(1-D)$ is the **theoretically correct** zero-sum formulation — gives clean 
  Nash equilibrium and JS divergence interpretation
- $-\log D$ is a **practical heuristic** that breaks zero-sum structure but trains 
  better — no need to switch back as generator improves since gradients naturally 
  diminish when $D(G(\mathbf{z})) \to \frac{1}{2}$

> This is a recurring theme in deep learning — theoretically correct formulations 
> are sometimes abandoned for pragmatic ones. WGAN later resolved this properly 
> by changing the objective entirely.

### Problem II: Non-Convergence

Gradient descent is not designed for min-max problems. Alternating GD may 
**orbit** rather than converge to the Nash equilibrium.

### Problem III: Mode Collapse

Generated samples cluster around one mode (low diversity).

**Fix — Mini-Batch GAN:** Feed batch statistics (e.g. pairwise $L^2$ norms, 
variance) into the discriminator so it can **reject low-diversity batches**.

## 4. Wasserstein GAN (WGAN)

### Why KL/JS Fail on Singular Distributions

When $p_G$ and $p^*$ have **disjoint support** (no overlap):
- KL = $\infty$, JS = $\log 2$ — both are **constants** w.r.t. $\theta$
- Zero gradient → training breaks down completely

**Parallel lines example:** $p_\theta =$ distribution of $(\theta, z)$, 
$p^* =$ distribution of $(0, z)$:

$$D_{KL}(p_\theta \| p^*) = \mathbb{1}_{\theta \neq 0}(+\infty), \qquad D_{JS}(p_\theta \| p^*) = \mathbb{1}_{\theta \neq 0}(\log 2)$$

Both are **discontinuous** at $\theta=0$ — gradient descent breaks down entirely.

### Wasserstein (Earth-Mover's) Distance

Measures the **minimum cost of transporting** mass from $p_G$ to $p^*$ — 
like the minimum effort to move a pile of dirt into a target shape:

$$W(p, q) = \inf_{\gamma \in \Pi(p,q)} \mathbb{E}_{(\mathbf{x},\mathbf{y})\sim\gamma}[\|\mathbf{x} - \mathbf{y}\|]$$

where $\Pi(p,q)$ = all joint distributions with marginals $p$ and $q$.

**Advantages over KL/JS:**

| Property | KL/JS | Wasserstein |
|---|---|---|
| Disjoint support | $\infty$ or constant | Smooth, finite |
| Gradient at $\theta=0$ | Undefined | $\pm 1$ — always exists |
| Symmetry | KL: No, JS: Yes | Yes |
| Parallel lines | Discontinuous | $W = \|\theta\|$ — continuous |

> **Key insight:** Wasserstein respects the **geometry of the space** — accounts 
> for how far apart distributions are, not just whether they overlap. KL/JS are 
> blind to distance when supports are disjoint.

### Kantorovich–Rubinstein Duality

$$W(p_{\boldsymbol{\theta}}, p^*) = \sup_{\|f\|_L \leq 1}\ \mathbb{E}_{\mathbf{x}\sim p_{\boldsymbol{\theta}}}[f(\mathbf{x})] - \mathbb{E}_{\mathbf{x}\sim p^*}[f(\mathbf{x})]$$

$\|f\|_L$ = Lipschitz constant of $f$ (think: $\max\|\nabla f\|$).

### WGAN Approximation & Training

Approximate $W$ using a neural **critic** $f(\mathbf{x}; \boldsymbol{\phi})$ 
constrained to be 1-Lipschitz (via **weight clipping** or **gradient penalty**):

$$W(p_{\boldsymbol{\theta}}, p^*) \approx \max_{\boldsymbol{\phi}}\ \mathbb{E}_{\mathbf{x}\sim p_{\boldsymbol{\theta}}}[f(\mathbf{x};\boldsymbol{\phi})] - \mathbb{E}_{\mathbf{x}\sim p^*}[f(\mathbf{x};\boldsymbol{\phi})]$$

**Gradient for generator:**
$$\nabla_{\boldsymbol{\theta}} W = \mathbb{E}_{\mathbf{z}\sim p_0}[\nabla_{\boldsymbol{\theta}} f(G_{\boldsymbol{\theta}}(\mathbf{z}); \hat{\boldsymbol{\phi}})]$$

### WGAN Algorithm

```
Repeat:
  For k steps:
    Sample {z^(i)} ~ p0, real {x^(i)} from D
    Update critic (ascent):
      φ ← φ + ε₁ (1/m) Σ ∇_φ [f(x^(i);φ) − f(G_θ(z^(i));φ)]
  
  Sample {z^(i)} ~ p0
  Update generator (descent):
    θ ← θ − ε₂ (1/m) Σ ∇_θ f(G_θ(z^(i)); φ)
```

## 5. Neural Style Transfer (NST)

### Goal

Given content image $\mathbf{x}_c$ and style image $\mathbf{x}_s$, 
**optimise a new image** $\mathbf{x}$ (not model weights):

$$\min_{\mathbf{x}}\ \alpha\, L_c(\mathbf{x}, \mathbf{x}_c) + \beta\, L_s(\mathbf{x}, \mathbf{x}_s)$$

### CNN Feature Notation

For a pre-trained $L$-layer CNN:
$$h^{(\ell)}_{ij}(\mathbf{x}) = j\text{-th pixel of }i\text{-th channel at layer }\ell$$

Layer hierarchy: edges (Conv1) → textures (Conv3) → object parts (Conv5).

### Content Loss

Matches **where** features appear — use deeper layers for semantic content:

$$E_c(\mathbf{x}, \mathbf{x}_c, \ell) = \frac{1}{2}\sum_{i,j}\left(h^{(\ell)}_{ij}(\mathbf{x}) - h^{(\ell)}_{ij}(\mathbf{x}_c)\right)^2$$

$$L_c(\mathbf{x}, \mathbf{x}_c) = \sum_{\ell \in \ell_c} E_c(\mathbf{x}, \mathbf{x}_c, \ell)$$

Use **deeper layers** (e.g. Conv4–5) to capture semantic content.

### Style Loss — Gram Matrix

The **Gram matrix** captures cross-channel feature correlations — 
*which features co-occur* regardless of *where* they appear:

$$G^{(\ell)}_{ij}(\mathbf{x}) = \sum_k h^{(\ell)}_{ik}(\mathbf{x})\, h^{(\ell)}_{jk}(\mathbf{x})$$

**Why Gram matrix captures style not content:**

| | Content | Style |
|---|---|---|
| **What matters** | *What* features appear *where* | *Which* features co-occur |
| **Spatial info** | Required | Discarded (summed over $k$) |
| **Tool** | Direct feature matching | Gram matrix |
| **CNN layers** | Deep (semantic) | Shallow (texture) |

> **Intuition:** Van Gogh's swirly style means certain texture features always 
> appear together. The Gram matrix captures these co-occurrence statistics — 
> it doesn't care *where* the swirls are, only *that* they co-occur.

**Layer-wise style loss:**
$$E_s(\mathbf{x}, \mathbf{x}_s, \ell) = \frac{1}{4N_l M_l}\sum_{i,j}\left(G^{(\ell)}_{ij}(\mathbf{x}) - G^{(\ell)}_{ij}(\mathbf{x}_s)\right)^2$$

$$L_s(\mathbf{x}, \mathbf{x}_s) = \sum_{\ell \in \ell_s} E_s(\mathbf{x}, \mathbf{x}_s; \ell)$$

where $N_l$ = number of channels, $M_l$ = spatial size at layer $\ell$.

## 6. Quick Comparison

| | GAN | WGAN | NST |
|---|---|---|---|
| **Objective** | Min-max JS divergence | Min Wasserstein distance | Min content + style loss |
| **Adversary role** | Discriminator ($D$) | Critic ($f$, unbounded output) | None (optimise image directly) |
| **Key problem** | Mode collapse, vanishing grad | Requires Lipschitz constraint | Choice of layers $\ell_c, \ell_s$ |
| **Loss for generator** | $\log(1-D)$ theory, $-\log D$ practice | $-\mathbb{E}[f(G(\mathbf{z}))]$ | Content + style losses |