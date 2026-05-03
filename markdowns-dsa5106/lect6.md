# Lecture 6 — Regularization

## 1. Regularization — Core Idea

> "Any modification we make to a learning algorithm that is intended to **reduce generalization error** but not its training error."

- Trade **some bias** to lower **variance**
- Modifies empirical risk minimization (ERM):

$$\tilde{R}(\boldsymbol{\theta}; X, \mathbf{y}) = R(\boldsymbol{\theta}; X, \mathbf{y}) + \alpha\,\Omega(\boldsymbol{\theta})$$

| Symbol | Meaning |
|--------|---------|
| $\Omega(\boldsymbol{\theta})$ | Regularizer |
| $\alpha \geq 0$ | Regularization strength |

## 2. $L^2$ Regularization (Weight Decay)

$$\Omega(\boldsymbol{\theta}) = \frac{1}{2}\|\boldsymbol{\theta}\|^2 = \frac{1}{2}\sum_i \theta_i^2$$

$$\tilde{R}(\boldsymbol{\theta}) = R(\boldsymbol{\theta}) + \frac{1}{2}\alpha\|\boldsymbol{\theta}\|^2$$

### Ridge Regression (linear model)

$$\hat{\mathbf{w}} = (X^TX + \alpha I)^{-1}X^T\mathbf{y}$$

### Effect via Eigendecomposition

Let $H = X^TX$ with eigenvectors $\mathbf{u}_i$ and eigenvalues $\lambda_i$. Expand $X^T\mathbf{y} = \sum_i \beta_i \mathbf{u}_i$:

| | Solution |
|--|--|
| Unregularized | $\hat{\mathbf{w}} = \sum_i \dfrac{\beta_i}{\lambda_i}\mathbf{u}_i$ |
| $L^2$-Regularized | $\hat{\mathbf{w}} = \sum_i \dfrac{\beta_i}{\lambda_i + \alpha}\mathbf{u}_i$ |

- If $\lambda_i \gg \alpha$: almost **no change**
- If $\lambda_i \ll \alpha$: data influence **removed** in that direction
- **Key insight:** regularization suppresses directions of small variance in the data

### Weight Decay Interpretation (GD)

$$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \epsilon\nabla R(\boldsymbol{\theta}_k) - \underbrace{\epsilon\alpha\boldsymbol{\theta}_k}_{\text{decay}}$$

If $R = \text{const}$: $\;\boldsymbol{\theta}_{k+1} = (1 - \epsilon\alpha)\boldsymbol{\theta}_k$

### Bayesian Interpretation
$L^2$ ↔ **Gaussian prior**: $p(\boldsymbol{\theta}) \propto \exp(-\alpha\|\boldsymbol{\theta}\|^2/2)$

## 3. $L^1$ Regularization (LASSO)

$$\Omega(\boldsymbol{\theta}) = \|\boldsymbol{\theta}\|_1 = \sum_i |\theta_i|$$

### Solution (diagonal quadratic loss)

$$\hat{\theta}_i = \begin{cases} \theta_i^* - \text{Sign}(\theta_i^*)\dfrac{\alpha}{\lambda_i} & |\theta_i^*| > \dfrac{\alpha}{\lambda_i} \\ 0 & |\theta_i^*| \leq \dfrac{\alpha}{\lambda_i} \end{cases}$$

- **Sparsity-inducing**: small weights are set to **exactly 0**
- Useful for **feature selection**

### Bayesian Interpretation
$L^1$ ↔ **Laplace prior**: $p(\boldsymbol{\theta}) \propto \exp(-\alpha\|\boldsymbol{\theta}\|_1)$

## 4. $L^1$ vs $L^2$ Comparison

| Property | $L^2$ | $L^1$ |
|----------|-------|-------|
| Effect | Shrinks weights proportionally | Sets small weights to **zero** |
| Sparsity | No | **Yes** |
| Feature selection | No | **Yes** |
| Prior | Gaussian | Laplace |
| Name | Ridge / Weight Decay | LASSO |
| Best when | All features relevant, correlated | Few features relevant |

> **Elastic Net** = combines both $L^1$ and $L^2$

### Neural Network Notes
- Regularize **weights** $W$ only, **not biases** $\mathbf{b}$ (TensorFlow default)
- Can use different $\alpha$ per layer

### L1 vs L2 Geometry

| | $L^2$ (Ridge) | $L^1$ (LASSO) |
|---|---|---|
| Shape | Smooth ball — no corners | Diamond — corners on axes |
| Effect | Shrinks weights proportionally | Sets small weights to exactly zero |
| Sparsity | No | Yes |
| Prior | Gaussian | Laplace |

**Why $L^1$ produces sparsity:**
Loss contours intersecting the $L^1$ diamond almost always hit a **corner first**
— corners sit exactly on the axes, forcing some weights to exactly zero.
$L^2$'s smooth ball has no corners — contours intersect at a point where all 
weights are nonzero, just shrunk.

## 5. Early Stopping

Monitor **validation loss**; stop when it stops improving.

**Algorithm (patience $p$):** Every $n$ steps, compute validation error. If improved → save $\boldsymbol{\theta}^*$, reset counter. Else → increment counter. Stop when counter $= p$.

### Variants

| Strategy | Description |
|----------|-------------|
| Variant I | Find optimal $\tau$ on subtrain/valid split → retrain on full data for $\tau$ steps |
| Variant II | Use early stopping loss value $\epsilon$ as stopping criterion on full data |

### Equivalence to $L^2$ Regularization

For a quadratic loss $R(\theta) = \frac{1}{2}\lambda(\theta - \theta^*)^2$, GD gives:

$$\theta_k = (1-\epsilon\lambda)^k\theta_0 + \left(1 - (1-\epsilon\lambda)^k\right)\theta^*$$

Early stopping at $\tau$ is a **weighted interpolation** between $\theta_0$ and is **equivalent** to $L^2$ regularization with:

$$\alpha = \frac{\lambda(1-\epsilon\lambda)^\tau}{1-(1-\epsilon\lambda)^\tau}$$

| Method | Implicit Penalty | Pulls Toward |
|---|---|---|
| Early stopping | $\frac{1}{2}\alpha(\theta - \theta_0)^2$ — distance from init | Initialization $\theta_0$ |
| $L^2$ regularization | $\frac{1}{2}\alpha\|\theta\|^2$ — magnitude of weights | Origin $\mathbf{0}$ |

> They are equivalent **only when** $\theta_0 = \mathbf{0}$ (zero initialization).
> Otherwise they differ in what they penalize.

- The implicit penalty is: $\;\frac{1}{2}\alpha(\theta - \theta_0)^2$ — distance from **initialization**, not origin.
- **Advantage:** Implicit regularization — no need to tune $\alpha$!

## 6. Adding Noise

> Rationale: model should be **robust to noise**

### Noise on Inputs

- Heuristic prior: output is insensitive to small input perturbations
- For linear regression with $\tilde{X} = X + Z$, $\mathbf{z}^{(i)} \sim \mathcal{N}(0, \delta I)$:

$$\mathbb{E}_Z\tilde{R}(\mathbf{w}) = R(\mathbf{w}) + \frac{1}{2}\delta N\|\mathbf{w}\|^2$$

→ Equivalent to **$L^2$ regularization** (heuristically true for nonlinear models too)

> Inductive bias is essentially the set of assumptions a model makes about the data in order to generalize beyond what it has seen during training.

> **Inductive bias:** $f^*$ should be smooth and insensitive to small 
> input perturbations — robustness is baked into training.

### Noise on Outputs — Label Smoothing

For $K$-class one-hot labels, replace:

$$1 \mapsto 1 - \alpha, \quad 0 \mapsto \frac{\alpha}{K-1}$$

Prevents the model from being overconfident.

> **Inductive bias:** No class should ever be predicted with absolute 
> certainty — the model should remain calibrated.

### Noise on Weights

Add perturbation $\delta\boldsymbol{\phi} \sim \mathcal{N}(0, I)$ to $\boldsymbol{\theta}$ during training:

$$\mathbb{E}_\phi\tilde{R}(\boldsymbol{\theta}) = \frac{1}{2}(f(\mathbf{x},\boldsymbol{\theta}) - y)^2 + \frac{m\delta^2}{2}\|\nabla_\theta f(\mathbf{x},\boldsymbol{\theta})\|^2 + \cdots$$

→ Penalizes sensitivity of predictions to weight perturbations

## Summary Table

| Method | Category | Key Effect | Equivalent To |
|--------|----------|-----------|---------------|
| $L^2$ (Ridge) | Training | Shrink weights; suppress low-variance directions | Gaussian prior |
| $L^1$ (LASSO) | Training | Sparse weights; feature selection | Laplace prior |
| Early Stopping | Training | Constrain distance from $\boldsymbol{\theta}_0$ | $L^2$ (linear case) |
| Input Noise | Training/Data | Robustness to input perturbations | $L^2$ (linear case) |
| Label Smoothing | Data | Prevent overconfidence | Output noise |
| Weight Noise | Training | Penalize weight sensitivity | — |

> **Note:** These methods are equivalent for linear models under specific assumptions, but **not equivalent** for nonlinear neural networks in general.
