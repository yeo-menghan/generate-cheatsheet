# Lecture 7 — Model, Data & Training Techniques

## 1. Model Ensembling

### Bagging (Bootstrap Aggregating)
Train $m$ models $f_1, \ldots, f_m$ on **random subsamples** of training data, then aggregate:

- **Regression:** $\bar{f}(x) = \dfrac{1}{m} \sum_{j=1}^{m} f_j(x)$
- **Classification:** $\bar{f}(x) = \text{Mode}\{f_j(x) : j = 1, \ldots, m\}$

**Why it works:** If each model has noise $\epsilon_i$ with $\mathbb{E}\epsilon_i = 0$, $\mathbb{E}\epsilon_i^2 = \sigma^2$, and errors are uncorrelated, then:
$$\bar{E}(x) = \frac{1}{m} E(x)$$
The ensemble error is $1/m$ of the individual model error. *(Key unrealistic assumption: uncorrelated errors)*

### Dropout
Efficient **approximation** of model ensembling — avoids storing $m$ separate models.

**Key idea:** Stochastic approximation $\dfrac{1}{m}\sum_i f_i = \mathbb{E}_\gamma f_\gamma$, 
sample $\gamma \sim \text{Uniform}\{1,\ldots,m\}$ each SGD step.

**Mechanism:** Apply a random binary mask $\boldsymbol{\mu}$ element-wise:
$$f(\mathbf{x}; W, \mathbf{b}, \boldsymbol{\mu}) = \sigma\!\left(W(\boldsymbol{\mu} \circ \mathbf{x}) + \mathbf{b}\right)$$

- Each coordinate of $\boldsymbol{\mu}$ is drawn iid: $P(\mu_i = 1) = p$ (keep probability), 
  $P(\mu_i = 0) = 1-p$ (drop rate)
- Since $\mathbb{E}[\boldsymbol{\mu}] = p\mathbf{1}$, weights are effectively shrunk by $p$
- **Ensemble size:** $2^m$ distinct subnetworks (share parameters $\boldsymbol{\theta}$)

**Why $2^m$ subnetworks?** Each of the $m$ neurons can independently be kept or 
dropped → $2^m$ possible binary masks $\boldsymbol{\mu}$.

**Weight Scaling Inference Rule:** At test time, set $\boldsymbol{\mu} = \mathbf{1}$ 
and replace $W \mapsto p \times W$.

**Why scaling is necessary:** During training, each neuron is active with probability 
$p$, so the expected input to the next layer is scaled by $p$. At inference with all 
neurons active, inputs are suddenly larger by $\frac{1}{p}$ — breaking the statistics 
the network was trained on. Multiplying weights by $p$ restores the same expected 
activation magnitude.

**Regularization connection:** For linear regression, dropout $\equiv$ weighted 
$L^2$ regularization:
$$\tilde{R}(\mathbf{w}) = \frac{1}{2}\|X\mathbf{w} - \mathbf{y}\|^2 + \frac{1}{2}\mathbf{w}^T Q(X, p)\mathbf{w}$$

## 2. Batch Normalization (BN)

### Motivation
In deep networks, weight magnitudes across layers affect gradient descent stability. 
The $\mathcal{O}(\epsilon^2)$ terms in the loss expansion become large when layer 
weight products are large, making learning rate selection very hard.

**Goal:** Normalize values at each layer so statistics are consistent across layers.

### BN Layer Definition
Given batch $H = \{\mathbf{h}^{(1)}, \ldots, \mathbf{h}^{(B)}\}$:

$$\text{BN}(H;\boldsymbol{\gamma},\boldsymbol{\beta})^{(i)} = \boldsymbol{\gamma} \circ \left(\frac{\mathbf{h}^{(i)} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}\right) + \boldsymbol{\beta}$$

where:
$$\boldsymbol{\mu} = \frac{1}{B}\sum_j \mathbf{h}^{(j)}, \qquad \boldsymbol{\sigma} = \sqrt{\frac{1}{B}\sum_j (\mathbf{h}^{(j)} - \boldsymbol{\mu})^2}$$

$\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^d$ are **learnable** scale and shift parameters.

### Two Steps

| Step | Formula | Purpose |
|------|---------|---------|
| Normalize | $\tilde{\mathbf{h}}^{(i)} = \dfrac{\mathbf{h}^{(i)} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$ | Zero mean, unit std within batch |
| Rescale | $\hat{\mathbf{h}}^{(i)} = \boldsymbol{\gamma} \circ \tilde{\mathbf{h}}^{(i)} + \boldsymbol{\beta}$ | Restore expressive power |

**Why both steps are necessary:**
- **Normalization** controls the *scale of computation* — stabilizes training by 
  ensuring consistent statistics across layers regardless of upstream weight magnitudes
- **Rescaling** restores *representational freedom* — forcing unit statistics is too 
  restrictive; the optimal representation for a layer may genuinely need a different 
  mean or variance. $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$ allow the network 
  to learn any mean and variance, but in a stable controlled way

### Usage in FC Networks
$$H_k' = H_k W + \mathbf{b} \;\longrightarrow\; \hat{H}_k = \text{BN}(H_k'; \boldsymbol{\gamma}, \boldsymbol{\beta}) \;\longrightarrow\; H_{k+1} = \text{ReLU}(\hat{H}_k)$$

### Training vs Inference
- **Training:** Use batch statistics $\boldsymbol{\mu}, \boldsymbol{\sigma}$
- **Inference:** Use **moving averages** of $\boldsymbol{\mu}, \boldsymbol{\sigma}$ 
  accumulated during training

### Notes
- BN **couples different samples** in a batch — first architecture to do so
- For CNNs: share $\boldsymbol{\beta}, \boldsymbol{\gamma}$ per channel (length = 
  #channels), applied across all spatial positions
- Bias $\mathbf{b}$ is redundant when BN is applied: $\text{BN}(XW + b) = \text{BN}(XW)$ 
  (absorbed into $\boldsymbol{\beta}$)

## 3. Data Augmentation

### Motivation — Generalization Gap
Statistical learning theory gives:
$$\left|R_{\text{emp}}^{(N)}(\boldsymbol{\theta}) - R_{\text{pop}}(\boldsymbol{\theta})\right| \leq \frac{\text{Complexity of Model}}{N}$$

Two ways to reduce: **(1) regularization** (reduce complexity), **(2) data augmentation** 
(increase $N$).

### Principle
Given augmentation $\mathbf{x} \mapsto g(\mathbf{x}; \boldsymbol{\xi})$:
$$f^*\!\left(g(\mathbf{x}; \boldsymbol{\xi})\right) \approx f^*(\mathbf{x})$$

Data augmentation encodes **prior knowledge** (inductive bias): the oracle $f^*$ is 
assumed invariant under $g(\cdot;\boldsymbol{\xi})$.

> **Inductive bias:** You are asserting that the label should be preserved under 
> the transformation. Always ask: *"Does this transformation change the label?"* 
> If yes — don't use it.

| Augmentation | Implicit Assumption |
|---|---|
| Random translations | $f^*$ is translation-invariant |
| Random rotations | $f^*$ is rotation-invariant |
| Random contrast/saturation | $f^*$ is lighting-invariant |

> ⚠️ **Caution:** Arbitrary rotations break MNIST labels (e.g., 6 → 9). 
> Must respect problem structure.

## 4. Learning Rate Decay

### Problem with Constant Learning Rate in SGD
For a quadratic problem:
$$\mathbb{E}R(\theta_k) \approx \underbrace{R(\theta_0)(1-\epsilon)^{2k}}_{\text{exponential convergence}} + \underbrace{\epsilon\!\left(1-(1-\epsilon)^{2k}\right)}_{\mathcal{O}(\epsilon)\text{ fluctuations}}$$

A constant $\epsilon$ leaves residual noise — near the minimum, the true gradient 
is small but stochastic noise isn't, causing the update to keep **overshooting** 
around the minimum rather than settling.

Solution: use a **decaying** schedule $\epsilon = \epsilon_k$.

### Theoretical Sufficient Conditions (Robbins-Monro)
$$\sum_{k=0}^{\infty} \epsilon_k = +\infty \quad \text{and} \quad \sum_{k=0}^{\infty} \epsilon_k^2 < +\infty$$

| Condition | Purpose | What breaks without it |
|---|---|---|
| $\sum \epsilon_k = \infty$ | Steps large enough to reach minimum | Decay too fast → freeze before converging |
| $\sum \epsilon_k^2 < \infty$ | Steps small enough to kill noise | Decay too slow → residual noise never dies |

> Together: decay **slow enough** to make progress, **fast enough** to eliminate noise.

Example: $\epsilon_k = \dfrac{a}{1+bk}$ ($\mathcal{O}(k^{-1})$ decay). Satisfies both 
conditions but rarely used in practice — slows convergence.

### Exponential Schedule (Common in Practice)
Parameters: initial rate $\epsilon_0$, decay interval $k_0$, decay ratio $\gamma \in (0,1)$:
$$\epsilon_k = \epsilon_0 \, \gamma^{\lfloor k/k_0 \rfloor}$$

## 5. Adversarial Examples & Adversarial Training

### Adversarial Example
For a trained classifier $\hat{f}$ with parameters $\hat{\boldsymbol{\theta}}$, 
an adversarial example solves:
$$\mathbf{x}' = \arg\max_{\mathbf{z},\, \|\mathbf{z}-\mathbf{x}\| \leq \delta} L\!\left(\hat{y}(\mathbf{z};\hat{\boldsymbol{\theta}}), y\right)$$

Finds the **worst-case small perturbation** that maximally increases loss — 
imperceptible to humans yet fools the model.

### Adversarial Training (Mini-Max)
$$\min_{\boldsymbol{\theta}} \frac{1}{N} \sum_{i=1}^{N} \max_{\mathbf{z}_i:\, \|\mathbf{x}_i - \mathbf{z}_i\| \leq \delta} L\!\left(\hat{y}(\mathbf{z}_i; \boldsymbol{\theta}), y_i\right)$$

**Inductive bias:** The decision boundary should be robust within an $\ell_\infty$ 
ball of radius $\delta$ around every training point — not just correct on the 
training points themselves. Forces the model's notion of similarity to align with 
human perception.

### FGSM Algorithm (Fast Gradient Sign Method)
**Hyperparameters:** $\delta$ (adversarial budget), $J$ (inner steps), $\epsilon_1, \epsilon_2$ (learning rates)

```
For k = 0, 1, ... do:
    z_0 = x
    For j = 0, ..., J-1 do:
        z_{j+1} = z_j + ε₂ · Sign(∇_z L(ŷ(z_j; θ_k), y))   ← ascent on input
    θ_{k+1} = θ_k − ε₁ · ∇_θ L(ŷ(z_J; θ_k), y)              ← descent on params
```

**Effect:** Forces the model to be robust within an $\ell_\infty$ ball of radius $\delta$ around each training point.

## Summary Table

| Technique | Category | What it does |
|-----------|----------|-------------|
| Bagging | Model/Training | Averages multiple models to reduce variance |
| Dropout | Model/Training | Stochastic ensemble via random masking; implicit $L^2$ reg |
| Batch Normalization | Model | Normalizes layer activations; stabilizes training |
| Data Augmentation | Data | Expands dataset using label-preserving transforms |
| Learning Rate Decay | Training | Reduces SGD noise as training converges |
| Adversarial Training | Training | Trains on worst-case perturbations for robustness |