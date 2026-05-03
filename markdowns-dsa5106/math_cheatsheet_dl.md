# Math Fundamentals for Deep Learning

## 1. Exponentials & Logarithms

### Core Rules

| Rule | Formula |
|------|---------|
| Product | $e^a \cdot e^b = e^{a+b}$ |
| Quotient | $e^a / e^b = e^{a-b}$ |
| Power | $(e^a)^b = e^{ab}$ |
| Log product | $\log(ab) = \log a + \log b$ |
| Log quotient | $\log(a/b) = \log a - \log b$ |
| Log power | $\log(a^b) = b\log a$ |
| Inverse | $\log(e^x) = x$, $\quad e^{\log x} = x$ |
| Change of base | $\log_b a = \dfrac{\ln a}{\ln b}$ |

### Key Values
$$e^0 = 1, \quad e^1 \approx 2.718, \quad \log 1 = 0, \quad \log 0 = -\infty$$

### Softmax Connection
$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$
Numerically stable version: subtract $\max_j z_j$ before exponentiating.

### Log-Sum-Exp Trick
$$\log \sum_j e^{z_j} = c + \log \sum_j e^{z_j - c}, \quad c = \max_j z_j$$
Avoids overflow/underflow; used in cross-entropy implementations.

## 2. Differentiation

### Basic Rules

| Rule | Formula |
|------|---------|
| Power | $\dfrac{d}{dx} x^n = nx^{n-1}$ |
| Exponential | $\dfrac{d}{dx} e^x = e^x$ |
| Exponential (chain) | $\dfrac{d}{dx} e^{f(x)} = f'(x)\,e^{f(x)}$ |
| Log | $\dfrac{d}{dx} \ln x = \dfrac{1}{x}$ |
| Chain rule | $\dfrac{dz}{dx} = \dfrac{dz}{dy}\dfrac{dy}{dx}$ |
| Product rule | $\dfrac{d}{dx}[f \cdot g] = f'g + fg'$ |
| Quotient rule | $\dfrac{d}{dx}\!\left[\dfrac{f}{g}\right] = \dfrac{f'g - fg'}{g^2}$ |

### Activation Function Derivatives

| Activation | $g(z)$ | $g'(z)$ |
|------------|--------|---------|
| Sigmoid | $\sigma(z) = \dfrac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z))$ |
| Tanh | $\tanh(z)$ | $1 - \tanh^2(z)$ |
| ReLU | $\max(0,z)$ | $\mathbf{1}[z > 0]$ |
| Leaky ReLU | $z$ if $z\geq0$, else $\delta z$ | $1$ if $z\geq0$, else $\delta$ |
| Softmax $k$ | $\dfrac{e^{z_k}}{\sum_j e^{z_j}}$ | $s_k(1 - s_k)$ (diagonal); $-s_i s_j$ (off-diagonal) |

> **Sigmoid saturation:** $\sigma'(z) \to 0$ as $|z| \to \infty$ — causes **vanishing gradients** in deep networks.

### Multivariate Chain Rule (Backprop)

For $f: \mathbb{R}^m \to \mathbb{R}^n$, $g: \mathbb{R}^n \to \mathbb{R}$:

$$\nabla_{\mathbf{x}} z = \left(\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\right)^{\!T} \nabla_{\mathbf{y}} z$$

where $\dfrac{\partial \mathbf{y}}{\partial \mathbf{x}}$ is the **Jacobian** ($n \times m$ matrix).

### Gradient of Common Loss Functions

| Loss | $L(y, \hat{y})$ | $\partial L / \partial \hat{y}$ |
|------|----------------|-------------------------------|
| MSE | $\tfrac{1}{2}(y - \hat{y})^2$ | $\hat{y} - y$ |
| Cross-entropy | $-\sum_k y_k \log \hat{y}_k$ | $-y_k / \hat{y}_k$ |
| BCE | $-[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ | $\dfrac{\hat{y}-y}{\hat{y}(1-\hat{y})}$ |

> **Softmax + cross-entropy:** gradient simplifies cleanly to $\hat{y}_k - y_k$.

## 3. Matrix & Vector Operations

### Notation

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x} \in \mathbb{R}^d$ | Column vector, $d$-dim |
| $A \in \mathbb{R}^{m \times n}$ | Matrix, $m$ rows, $n$ cols |
| $A^T$ | Transpose |
| $A^{-1}$ | Inverse (square, full-rank only) |
| $\|{\mathbf{x}}\|_2$ | L2 norm $= \sqrt{\sum_i x_i^2}$ |
| $\|{\mathbf{x}}\|_1$ | L1 norm $= \sum_i |x_i|$ |

### Key Identities

$$\mathbf{x}^T\mathbf{y} = \mathbf{y}^T\mathbf{x} \quad \text{(dot product, scalar)}$$
$$(AB)^T = B^T A^T$$
$$(AB)^{-1} = B^{-1}A^{-1}$$
$$(A^T)^{-1} = (A^{-1})^T$$

### Matrix Calculus (used in backprop)

| Expression | Gradient w.r.t. $\mathbf{w}$ |
|------------|-------------------------------|
| $\mathbf{w}^T\mathbf{x}$ | $\mathbf{x}$ |
| $\mathbf{x}^T A \mathbf{x}$ | $(A + A^T)\mathbf{x}$ |
| $\|\mathbf{x} - A\mathbf{w}\|^2$ | $-2A^T(\mathbf{x} - A\mathbf{w})$ |
| $\frac{1}{2}\|X\mathbf{w} - \mathbf{y}\|^2$ | $X^T(X\mathbf{w} - \mathbf{y})$ |

### OLS & Ridge Solutions

$$\hat{\mathbf{w}}_{\text{OLS}} = (X^TX)^{-1}X^T\mathbf{y}$$
$$\hat{\mathbf{w}}_{\text{Ridge}} = (X^TX + \alpha I)^{-1}X^T\mathbf{y}$$

### Eigendecomposition

For symmetric $A = U\Lambda U^T$:
- $U$ — orthogonal matrix of eigenvectors
- $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_d)$ — eigenvalues

$$A\mathbf{u}_i = \lambda_i \mathbf{u}_i, \qquad U^TU = I$$

**Condition number:** $\kappa(A) = \dfrac{\lambda_{\max}}{\lambda_{\min}}$ — large $\kappa$ → slow GD convergence.

### Norms Summary

| Norm | Formula | Use in DL |
|------|---------|-----------|
| L1 | $\|\mathbf{w}\|_1 = \sum_i \vert w_i \vert$ | LASSO, sparsity |
| L2 | $\|\mathbf{w}\|_2 = \sqrt{\sum_i w_i^2}$ | Ridge, weight decay |
| Frobenius | $\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2}$ | Matrix regularization |

## 4. Probability & Statistics Essentials

### Expectation & Variance

$$\mathbb{E}[X] = \sum_x x\,p(x) \quad \text{(discrete)}, \qquad \mathbb{E}[X] = \int x\,p(x)\,dx \quad \text{(continuous)}$$
$$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$
$$\text{Var}(aX + b) = a^2\,\text{Var}(X)$$

### Key Distributions

| Distribution | PDF/PMF | Mean | Variance |
|---|---|---|---|
| Gaussian $\mathcal{N}(\mu, \sigma^2)$ | $\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ |
| Bernoulli$(p)$ | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ |

### Bayes' Theorem
$$p(\boldsymbol{\theta} \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid \boldsymbol{\theta})\,p(\boldsymbol{\theta})}{p(\mathbf{x})}$$

| Term | Name |
|------|------|
| $p(\boldsymbol{\theta} \mid \mathbf{x})$ | Posterior |
| $p(\mathbf{x} \mid \boldsymbol{\theta})$ | Likelihood |
| $p(\boldsymbol{\theta})$ | Prior |
| $p(\mathbf{x})$ | Evidence (normalizer) |

### KL Divergence
$$D_{KL}(p \| q) = \mathbb{E}_{p}\!\left[\log\frac{p(x)}{q(x)}\right] \geq 0, \quad = 0 \iff p = q$$

> Not symmetric: $D_{KL}(p\|q) \neq D_{KL}(q\|p)$

## 5. Useful Miscellaneous Identities

### Taylor Expansion (1st order)
$$f(\mathbf{x} + \epsilon\boldsymbol{\phi}) \approx f(\mathbf{x}) + \epsilon\,\boldsymbol{\phi}^T\nabla f(\mathbf{x})$$
Used to derive gradient descent: choose $\boldsymbol{\phi} = -\nabla f$ to decrease $f$.

### Sigmoid Identities
$$\sigma(-z) = 1 - \sigma(z), \qquad \log \sigma(z) = -\log(1 + e^{-z})$$

### Log-Likelihood & Cross-Entropy Connection
$$-\log p(y|\mathbf{x}) = \text{cross-entropy}(y, \hat{y})$$
Minimising cross-entropy ≡ maximising log-likelihood.

### Dot Product & Cosine Similarity
$$\mathbf{a}^T\mathbf{b} = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta, \qquad \cos\theta = \frac{\mathbf{a}^T\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$$
Used in attention: $e_{ij} = q_i^T k_j$ is proportional to cosine similarity when vectors are normalized.

### Frobenius Inner Product
$$\langle A, B \rangle_F = \text{tr}(A^T B) = \sum_{i,j} A_{ij}B_{ij}$$

### Trace Tricks
$$\text{tr}(AB) = \text{tr}(BA), \qquad \mathbf{x}^T A \mathbf{x} = \text{tr}(A\mathbf{x}\mathbf{x}^T)$$

## 6. Confusion Matrix & Classification Metrics
 
### The Confusion Matrix (Binary Classification)
 
|  | Predicted Positive | Predicted Negative |
|--|---|---|
| **Actual Positive** | TP (True Positive) | FN (False Negative) |
| **Actual Negative** | FP (False Positive) | TN (True Negative) |
 
- **TP** — correctly predicted positive
- **TN** — correctly predicted negative
- **FP** — predicted positive, actually negative (Type I error)
- **FN** — predicted negative, actually positive (Type II error)
### Metrics
 
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
 
> Fraction of all predictions that were correct. **Misleading on imbalanced datasets** — a model predicting all negatives on a 99:1 dataset gets 99% accuracy while being useless.
 
$$\text{Precision} = \frac{TP}{TP + FP}$$
 
> Of all the times the model said "positive", how often was it right? High precision = **few false alarms**. Optimise this when false positives are costly (e.g. spam filter wrongly blocking legitimate email).
 
$$\text{Recall} = \frac{TP}{TP + FN}$$
 
> Of all actual positives, how many did the model catch? High recall = **few missed positives**. Optimise this when false negatives are costly (e.g. missing a cancer diagnosis).
 
$$\text{F1} = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$
 
> Harmonic mean of precision and recall — balances both. Preferred over accuracy on **imbalanced datasets**. The harmonic mean punishes extreme imbalance between the two: a model with precision 1.0 and recall 0.01 gets F1 = 0.02, not 0.5.
 
### Precision–Recall Tradeoff
 
Precision and recall are in **tension** — controlled by the classification threshold $\tau$:
 
| Threshold $\tau$ | Effect |
|---|---|
| $\tau$ high (strict) | Fewer positives predicted → Precision ↑, Recall ↓ |
| $\tau$ low (lenient) | More positives predicted → Recall ↑, Precision ↓ |
 
> **Intuition:** A spam filter set to block aggressively (low $\tau$) catches more spam (high recall) but also blocks more legitimate email (low precision). Loosening it does the opposite.
 
The **PR curve** plots Precision vs Recall across all thresholds. A model with a larger **area under the PR curve (AUC-PR)** is better, especially under class imbalance. A random classifier's AUC-PR equals the fraction of positives in the dataset.
  
## 7. Bias–Variance Tradeoff
 
### Decomposition
 
For a model $\hat{f}$ trained on dataset $\mathcal{D}$, the expected test error decomposes as:
 
$$\mathbb{E}\left[(y - \hat{f}(\mathbf{x}))^2\right] = \underbrace{\text{Bias}^2(\hat{f})}_{\text{underfitting}} + \underbrace{\text{Var}(\hat{f})}_{\text{overfitting}} + \underbrace{\sigma^2}_{\text{irreducible noise}}$$
 
| Term | Formula | Meaning |
|------|---------|---------|
| **Bias** | $\mathbb{E}[\hat{f}(\mathbf{x})] - f^*(\mathbf{x})$ | How far the average prediction is from the truth — error from wrong assumptions |
| **Variance** | $\mathbb{E}\left[(\hat{f}(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})])^2\right]$ | How much the model fluctuates across different training sets |
| **Noise** | $\sigma^2 = \text{Var}(y \mid \mathbf{x})$ | Irreducible — inherent randomness in the data |
 
### Intuition
 
| Scenario | Bias | Variance | Symptom |
|---|---|---|---|
| **Underfitting** | High | Low | Model too simple; misses true pattern |
| **Overfitting** | Low | High | Model too complex; memorises noise |
| **Good fit** | Low | Low | Generalises well |
 
> **Analogy:** Bias is a consistently off-centre archer (wrong aim). Variance is a scattered archer (inconsistent). You want both centred and tight grouping.
 
### Connection to Regularization & Model Capacity
 
- Increasing model capacity (more layers, more neurons) → **bias ↓, variance ↑**
- Adding regularization ($L^1$, $L^2$, dropout, early stopping) → **variance ↓, bias ↑ slightly**
- Adding more data → **variance ↓**, bias unchanged
> Regularization methods from Lecture 6 are all mechanisms to shift the bias–variance tradeoff toward lower variance at the cost of a small increase in bias.