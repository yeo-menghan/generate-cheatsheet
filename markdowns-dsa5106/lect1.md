# DSA5106 — Lecture 1

## 1. The T, E, P Framework (Tom Mitchell)

> *A program learns from **Experience E** w.r.t. **Task T** measured by **Performance P** if P improves with E.*

| Component | Meaning | Example |
|---|---|---|
| **T** (Task) | What to do | Predict house prices |
| **E** (Experience) | Training data $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ | Labelled dataset |
| **P** (Performance) | Metric | MSE, Accuracy |

## 2. Empirical Risk Minimization (ERM)

**Distance / Loss** over dataset:
$$\text{Distance} = \frac{1}{N}\sum_{i=1}^{N} L(f(\mathbf{x}_i),\, y_i)$$

This is called the **empirical risk** $R_\text{emp}(f, \mathcal{D})$.

**ERM:**
$$\hat{f} = \arg\min_{f \in \mathcal{H}}\; R_\text{emp}(f, \mathcal{D})$$

**Goal (population risk):**
$$\tilde{f} = \arg\min_{f \in \mathcal{H}}\; R_\text{pop}(f) = \mathbb{E}_{\mathbf{x}\sim\mu}\bigl[L(f(\mathbf{x}), f^*(\mathbf{x}))\bigr]$$

> ⚠️ ERM $\neq$ population risk minimization — the gap is the **generalization gap**.

## 3. Linear Regression

**Hypothesis space (affine):**
$$\mathcal{H} = \{f(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + b : \mathbf{w} \in \mathbb{R}^d,\, b \in \mathbb{R}\}$$

**Square loss:**
$$L(y', y) = \tfrac{1}{2}(y' - y)^2$$

**ERM (matrix form):**
$$\min_{\mathbf{w}}\; R_\text{emp}(\mathbf{w}) = \min_{\mathbf{w}}\; \frac{1}{2N}\|X\mathbf{w} - \mathbf{y}\|^2$$

**Ordinary Least Squares (OLS) solution:**
$$\hat{\mathbf{w}} = (X^T X)^{-1} X^T \mathbf{y}$$

*For affine models, append a column of 1s to $X$ and append $b$ to $\mathbf{w}$.*

## 4. Linear Basis (Feature) Models

When linear functions are insufficient, use **basis functions** $\boldsymbol{\phi} = (\phi_1,\ldots,\phi_m)$:

$$\mathcal{H} = \left\{f(\mathbf{x}) = \mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}) = \sum_{i=1}^m w_i \phi_i(\mathbf{x}) : \mathbf{w} \in \mathbb{R}^m\right\}$$

| Basis | Formula |
|---|---|
| Polynomial | $\phi_j(x) = x^j$ |
| Gaussian | $\phi_j(x) = \exp\!\left(-\frac{(x-m_j)^2}{2s^2}\right)$ |
| Sigmoid | $\phi_j(x) = \sigma\!\left(\frac{x-m_j}{s}\right),\quad \sigma(b)=\frac{1}{1+e^{-b}}$ |

**Solution:** Replace $X$ with feature matrix $\Phi$ (rows = $\boldsymbol{\phi}(\mathbf{x}_i)$):
$$\hat{\mathbf{w}} = (\Phi^T\Phi)^{-1}\Phi^T\mathbf{y}$$

**Universality:** With $m\to\infty$ and appropriate bases, linear basis models can approximate *any* function (Fourier series, Weierstrass approximation).

## 5. Model Capacity, Underfitting & Overfitting

| Issue | Cause | Symptom |
|---|---|---|
| **Underfitting** | $\mathcal{H}$ too small (low capacity) | High train & test error |
| **Overfitting** | $\mathcal{H}$ too large (high capacity) | Low train, high test error |

**Remedy:** Train-test split $\mathcal{D} = \mathcal{D}_\text{train} \cup \mathcal{D}_\text{test}$
- Train on $\mathcal{D}_\text{train}$, evaluate generalization on $\mathcal{D}_\text{test}$
- **Never** let the learning algorithm peek at $\mathcal{D}_\text{test}$

**Techniques to improve generalization:** architecture choice, regularization, data augmentation, optimization methods.

## 6. Classification

### Setup
- Labels encoded as **one-hot vectors**: $\mathbf{y}_i = (0,\ldots,1,\ldots,0) \in \mathbb{R}^K$
- Model outputs class probabilities: $f(\mathbf{x}_i) = (a_1,\ldots,a_K) \in \mathbb{R}^K$

### Softmax activation
$$s(\mathbf{z})_k = \frac{\exp(z_k)}{\sum_j \exp(z_j)}$$

**Hypothesis space:**
$$\mathcal{H} = \{f(\mathbf{x}) = s(W\boldsymbol{\phi}(\mathbf{x})) : W \in \mathbb{R}^{K\times m}\}$$

### Loss functions for classification

| Loss | Formula | Notes |
|---|---|---|
| **Zero-one** | $\mathbf{1}[\text{wrong}]$ | Not differentiable — can't use gradient descent |
| **MSE** | $\frac{1}{2}\|\mathbf{y} - \mathbf{y}'\|^2$ | Bounded gradient; less effective |
| **Cross-entropy** | $-\sum_{k=1}^K y_k' \log y_k$ | Preferred; large gradient when wrong |

> Use **cross-entropy** with softmax for classification — gradients don't vanish the way MSE does.

## 7. Key Concepts Summary

| Concept | Description |
|---|---|
| Hypothesis space $\mathcal{H}$ | Set of candidate functions |
| Capacity | Size / expressivity of $\mathcal{H}$ |
| Empirical risk | Average loss on training data |
| Population risk | Expected loss over the true distribution |
| Generalization gap | $R_\text{pop}(\hat{f}) - R_\text{emp}(\hat{f})$ |
| Basis functions / feature maps | Fixed nonlinear transforms of input |
| Softmax | Converts logits to probability distribution |
| Cross-entropy loss | Standard loss for classification |