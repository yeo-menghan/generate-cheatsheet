# Lecture 2

## 1. Motivation: Representation Learning

### From Linear Basis to Adaptive Models

| Model | Formula |
|-------|---------|
| Linear regression | $f(\mathbf{x}) = \mathbf{w}^T\mathbf{x}$ |
| Linear basis model | $f(\mathbf{x}) = \mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})$ |
| Adaptive basis model | $f(\mathbf{x}) = \mathbf{w}^T\boldsymbol{\phi}(\mathbf{x}; \boldsymbol{\theta})$ |

- Fixed basis $\boldsymbol{\phi}$ (polynomial, Fourier, Haar wavelet) works well for some $f^*$ but not all.
- **Key insight**: Adapt $\boldsymbol{\phi}$ to data by learning $\boldsymbol{\theta}$ — this is **representation learning**.

## 2. Why Not Just Use Large Linear Basis Models?

### Curse of Dimensionality

For a $d$-dimensional input, the number of polynomial basis functions grows **exponentially** with $d$.

### Approximation Rates

| Method | Error bound |
|--------|------------|
| Linear basis model (Jackson, 1912) | $\|f^* - f_m\|^2 \leq \mathcal{O}(m^{-2\alpha/d})$ |
| Neural network (Barron, 1993) | $\|f^* - f_m\|^2 \leq \mathcal{O}(m^{-1})$ |

where $m$ = number of neurons/basis functions, $\alpha$ = smoothness of $f^*$, $d$ = input dimension.

**Neural networks beat the curse of dimensionality** — their approximation rate is *independent* of $d$.

## 3. Neural Networks as Adaptive Basis Models

### Shallow (1-hidden-layer) Network

$$f(\mathbf{x}; W, \mathbf{c}, \mathbf{w}, b) = \mathbf{w}^T \underbrace{g(W\mathbf{x} + \mathbf{c})}_{\boldsymbol{\phi}(\mathbf{x};\boldsymbol{\theta})} + b$$

**Parameters:**
- $W \in \mathbb{R}^{m \times d}$ — input weight matrix
- $\mathbf{c} \in \mathbb{R}^m$ — hidden bias
- $\mathbf{w} \in \mathbb{R}^m$ — output weight
- $b \in \mathbb{R}$ — output bias
- $g$ — activation function (applied element-wise)

**Composition:** $f = f^{(2)} \circ f^{(1)}$ where

$$\mathbf{h} = f^{(1)}(\mathbf{x}; W, \mathbf{c}) = g(W\mathbf{x} + \mathbf{c}) \qquad \text{(hidden layer)}$$
$$y = f^{(2)}(\mathbf{h}; \mathbf{w}, b) = \mathbf{w}^T\mathbf{h} + b \qquad \text{(output layer)}$$

> **Why nonlinearity?** Composing linear layers collapses to a single linear model. Nonlinearity is essential.
> Consider $f(x) = W_2(W_1x + b_1) + b_2$ 

### XOR Example

A simple 2-layer ReLU network with

$$W = \begin{pmatrix}1&1\\1&1\end{pmatrix},\quad \mathbf{c} = \begin{pmatrix}0\\-1\end{pmatrix},\quad \mathbf{w} = \begin{pmatrix}1\\-2\end{pmatrix},\quad b=0$$

perfectly represents the XOR function.

## 4. Activation Functions

| Name | Formula |
|------|---------|
| **ReLU** | $g(z) = \max(0, z)$ |
| **Sigmoid** | $g(z) = \dfrac{1}{1+e^{-z}}$ |
| **Tanh** | $g(z) = \tanh(z)$ |
| **Leaky ReLU** | $g(z) = z$ if $z \geq 0$, else $\delta z$ |

Each neuron computes: $g(W_i \cdot \mathbf{x} + c_i)$

## 5. Universal Approximation Theorem

> Any continuous function $f^*$ on a compact domain can be approximated to arbitrary precision by a neural network, provided $m$ (number of neurons) is large enough.

## 6. Output Units & Loss Functions

### Regression
- **Output**: linear $\hat{y} = \mathbf{w}^T\mathbf{h} + b$
- **Loss**: Mean squared error (MSE) $L(y, y') = \tfrac{1}{2}(y - y')^2$

### Multi-class Classification
- **Output**: $\text{Softmax}(\mathbf{z})_i = \dfrac{\exp(z_i)}{\sum_j \exp(z_j)}$
- **Loss**: Cross-entropy $L(\mathbf{y}, \mathbf{y}') = -\sum_j y_j' \log y_j$

### Multi-label Classification
- **Output**: $\text{Sigmoid}(\mathbf{z})_i = \dfrac{1}{1+\exp(-z_i)}$
- **Loss**: Binary cross-entropy $L(\mathbf{y}, \mathbf{y}') = -\sum_j \left[y_j' \log y_j + (1-y_j')\log(1-y_j)\right]$

### Zero-One Loss (not used for optimization)
$$L(y, y') = \mathbb{1}_{y \neq y'} \quad \text{(non-differentiable)}$$

> **Cross-entropy vs MSE for classification**: Cross-entropy has better statistical interpretation (= log-likelihood maximization) and better numerical stability with softmax.

## 7. Gradient-Based Learning

### Empirical Risk Minimization (ERM)

$$\min_{\boldsymbol{\theta}} R_{\text{emp}}(\boldsymbol{\theta}) = \frac{1}{N}\sum_{i=1}^N L\bigl(f^*(\mathbf{x}_i),\, f(\mathbf{x}_i; \boldsymbol{\theta})\bigr)$$

### Gradient Descent (GD)

From Taylor expansion: $R(\boldsymbol{\theta} + \epsilon\boldsymbol{\phi}) \approx R(\boldsymbol{\theta}) + \epsilon\boldsymbol{\phi}^T\nabla R(\boldsymbol{\theta})$

To decrease $R$, choose $\boldsymbol{\phi} \propto -\nabla R(\boldsymbol{\theta})$:

$$\boxed{\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \epsilon \nabla R(\boldsymbol{\theta}_k)}$$

- $\epsilon$ = **learning rate / step size**
- $-\nabla R$ = **steepest descent direction**

### Convergence (Linear Regression Case)

For $\min_\mathbf{w} \frac{1}{2}\|X\mathbf{w} - \mathbf{y}\|^2$, GD converges if:

$$\epsilon < \frac{2}{\max_i \lambda_i(X^TX)}$$

Convergence rate depends on the **condition number**:
$$\kappa(X^TX) = \frac{\max_i \lambda_i(X^TX)}{\min_i \lambda_i(X^TX)} \geq 1$$

$$\|\mathbf{e}_k\| \leq \left(1 - \frac{1}{\kappa(X^TX)}\right)^k \|\mathbf{e}_0\|$$

Larger condition number → slower convergence.

### General Convergence
If $\nabla R$ is Lipschitz and $\nabla^2 R$ is bounded, then for small enough $\epsilon$:
$$\|\nabla R(\boldsymbol{\theta}_k)\| \to 0 \quad \text{as } k \to \infty$$

## 8. Summary Table

| Concept | Key Formula / Idea |
|---------|-------------------|
| Adaptive basis | $f(\mathbf{x}) = \mathbf{w}^T\boldsymbol{\phi}(\mathbf{x};\boldsymbol{\theta})$ — learn $\boldsymbol{\theta}$ from data |
| Shallow NN | $f = \mathbf{w}^T g(W\mathbf{x}+\mathbf{c}) + b$ |
| Hidden units | Learn a new feature representation $\mathbf{h} = g(W\mathbf{x}+\mathbf{c})$ |
| Universal approx. | Any continuous $f^*$ can be approximated with enough neurons |
| Overcomes CoD | NN error $\mathcal{O}(m^{-1})$ vs. linear basis $\mathcal{O}(m^{-2\alpha/d})$ |
| Gradient descent | $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \epsilon\nabla R(\boldsymbol{\theta})$ |
