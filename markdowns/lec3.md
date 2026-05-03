# Lecture 3: Logistic Regression & Regularization

## 1. Logistic Regression

### Sigmoid (Logistic) Function
$$g(z) = \frac{1}{1 + e^{-z}}, \quad 0 < g(z) < 1$$
- $g(0) = 0.5$, $g(z) \to 1$ as $z \to +\infty$, $g(z) \to 0$ as $z \to -\infty$

### Model
$$f(x) = g(\hat{\beta}^T \hat{x}), \quad \hat{\beta} = [\beta_0; \beta], \quad \hat{x} = [1; x]$$

### Probabilistic Interpretation
$$f(x) = P(y=1 \mid x;\hat{\beta}), \quad 1 - f(x) = P(y=0 \mid x;\hat{\beta})$$

### Prediction Rule
- Predict $y = 1$ if $f(x) \geq 0.5$, i.e., $\hat{\beta}^T\hat{x} \geq 0$
- Predict $y = 0$ if $f(x) < 0.5$, i.e., $\hat{\beta}^T\hat{x} < 0$

### Decision Boundary
$$\beta_0 + \beta^T x = 0$$
| Features $p$ | Boundary shape |
|---|---|
| 1 | Point |
| 2 | Line |
| 3 | Plane |
| general | $(p-1)$-dimensional subspace |

## 2. Cost Function (Negative Log-Likelihood)

### Per-sample cost
$$-y_i \log(f(x_i)) - (1-y_i)\log(1 - f(x_i)) = \begin{cases} -\log(f(x_i)) & \text{if } y_i=1 \\ -\log(1-f(x_i)) & \text{if } y_i=0 \end{cases}$$

### Total cost (simplified form)
$$L(\beta_0, \beta) = \sum_{i=1}^n \log(1 + e^{\beta_0 + \beta^T x_i}) - y_i(\beta_0 + \beta^T x_i)$$

## 3. Gradient of the Cost Function

$$\frac{\partial L}{\partial \beta_0} = \sum_{i=1}^n (f(x_i) - y_i)$$

$$\frac{\partial L}{\partial \beta_j} = \sum_{i=1}^n (f(x_i) - y_i)\, x_{ij}, \quad j = 1, \dots, p$$

> **Note:** Same form as linear regression gradient, but $f(x_i)$ is the sigmoid output.

## 4. Multi-class: One-vs-Rest

For $K$ classes, train $K$ binary classifiers $f_1, \dots, f_K$.  
Predict class $k^* = \arg\max_k f_k(x)$.

## 5. Regularization

### Ridge (L2)
$$\text{Regularizer: } \lambda\|\beta\|^2 = \lambda\sum_{j=1}^p \beta_j^2$$
- Differentiable; shrinks all $\beta_j$ toward 0 but rarely to exactly 0.

**Logistic + Ridge:**
$$\min \sum_{i=1}^n \log(1+e^{\beta_0+\beta^Tx_i}) - y_i(\beta_0+\beta^Tx_i) + \lambda\sum_{j=1}^p \beta_j^2$$

**Linear + Ridge (Normal Equation**, assuming $\beta_0 = 0$):
$$\min_\beta \frac{1}{2}\|X\beta - Y\|^2 + \lambda\|\beta\|^2$$
$$\Rightarrow \quad \beta = (2\lambda I + X^TX)^{-1}X^TY$$

### Lasso (L1)
$$\text{Regularizer: } \lambda\|\beta\|_1 = \lambda\sum_{j=1}^p |\beta_j|$$
- Non-differentiable; forces some $\beta_j$ to exactly **zero** → feature selection.
- Gradient descent **not applicable**; requires specialised solvers.

**Lasso problem:**
$$\min_{\beta \in \mathbb{R}^p} \frac{1}{2}\|X\beta - Y\|^2 + \lambda\|\beta\|_1$$

## 6. Ridge vs. Lasso Summary

| | Ridge | Lasso |
|---|---|---|
| Penalty | $\lambda\|\beta\|^2$ | $\lambda\|\beta\|_1$ |
| Differentiable | ✅ | ❌ |
| Shrinks $\beta_j$ | Toward 0 | To exactly 0 |
| Feature selection | ❌ | ✅ |
| Solver | Gradient descent / Normal eq. | Specialised (e.g. proximal) |

> **Larger $\lambda$** → stronger regularization → simpler model (risk of underfitting).  
> **Smaller $\lambda$** → weaker regularization → more complex model (risk of overfitting).
