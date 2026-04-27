# Lecture 2: Gradient Descent & Linear Regression

---

## 1. Gradient Descent Basics

**Stationary point:** $\nabla f(x) = 0$ — necessary condition for a local min.

**Key property of gradient (Proposition 1):**
- $\nabla f(x^*)^T d$ = rate of change of $f$ moving from $x^*$ in direction $d$
- $f$ **decreases most rapidly** along $-\nabla f(x^*)$ (steepest descent direction)
- $f$ **increases most rapidly** along $\nabla f(x^*)$

**General iterative update:**
$$x^{(k+1)} = x^{(k)} + \alpha_k p^{(k)}$$
where $\alpha_k > 0$ is the step length/learning rate and $p^{(k)}$ is the search direction.

---

## 2. Descent Direction

$p^{(k)}$ is a **descent direction** at $x^{(k)}$ if:
$$\nabla f(x^{(k)})^T p^{(k)} < 0$$

This guarantees $\exists\, \delta > 0$ such that $f(x^{(k)} + \alpha_k p^{(k)}) < f(x^{(k)})$ for all $\alpha_k \in (0, \delta)$.

> **Tip:** $p = -\nabla f(x^{(k)})$ always satisfies this since $\nabla f^T(-\nabla f) = -\|\nabla f\|^2 \leq 0$.

---

## 3. Steepest Descent Method

**Update rule:**
$$x^{(k+1)} = x^{(k)} - \alpha_k \nabla f(x^{(k)})$$

**Stopping criterion:** $\|\nabla f(x^{(k)})\| \leq \epsilon$

### Step Length Options

| Method | Description |
|--------|-------------|
| **Constant** | Fix $\alpha_k = c$. Too small → slow; too large → diverge |
| **Exact line search** | $\alpha_k = \arg\min_{\alpha>0} f(x^{(k)} + \alpha p^{(k)})$, solve $\phi'(\alpha)=0$ |
| **Backtracking** | Start large, shrink until Armijo condition holds |

### Exact Line Search
Solve: $\min_{\alpha > 0}\ \phi(\alpha) = f(x^{(k)} + \alpha p^{(k)})$

Set $\phi'(\alpha) = 0$ and solve for $\alpha_k$.

### Backtracking Line Search (Armijo)
Parameters: $\bar{\alpha} > 0,\ \rho \in (0,1),\ c_1 \in (0,1)$ (e.g. $\bar\alpha=1, \rho=0.9, c_1=10^{-4}$)

Shrink $\alpha \leftarrow \rho\alpha$ until:
$$f(x^{(k)} + \alpha p^{(k)}) \leq f(x^{(k)}) + c_1 \alpha \nabla f(x^{(k)})^T p^{(k)}$$

> Note: RHS < LHS initially since $\nabla f^T p < 0$, so the condition requires sufficient decrease.

---

## 4. Properties of Steepest Descent (Exact Line Search)

- **Perpendicular steps:** consecutive steps are orthogonal
- **Monotone decrease:** $f(x^{(k+1)}) < f(x^{(k)})$ if $\nabla f(x^{(k)}) \neq 0$
- **Zig-zag behaviour** near the solution (inherently slow)

---

## 5. Linear Regression — One Variable

**Model:** $f(x) = \beta_1 x + \beta_0$

**Cost function:**
$$L(\beta_0, \beta_1) = \frac{1}{2}\sum_{i=1}^n (\beta_1 x_i + \beta_0 - y_i)^2$$

**Gradients:**
$$\frac{\partial L}{\partial \beta_0} = \sum_{i=1}^n (\beta_1 x_i + \beta_0 - y_i)$$
$$\frac{\partial L}{\partial \beta_1} = \sum_{i=1}^n (\beta_1 x_i + \beta_0 - y_i)\, x_i$$

**Update rule:**
$$\beta_0^{(k+1)} = \beta_0^{(k)} - \alpha_k \sum_{i=1}^n (\beta_1^{(k)} x_i + \beta_0^{(k)} - y_i)$$
$$\beta_1^{(k+1)} = \beta_1^{(k)} - \alpha_k \sum_{i=1}^n (\beta_1^{(k)} x_i + \beta_0^{(k)} - y_i)\, x_i$$

---

## 6. Linear Regression — Multiple Variables

**Model:** $f(x) = \beta^T x + \beta_0,\quad \beta \in \mathbb{R}^p$

**Cost function:**
$$L(\beta_0, \beta) = \frac{1}{2}\sum_{i=1}^n (\beta^T x_i + \beta_0 - y_i)^2$$

**Gradients:**
$$\frac{\partial L}{\partial \beta_0} = \sum_{i=1}^n (\beta^T x_i + \beta_0 - y_i)$$
$$\frac{\partial L}{\partial \beta_j} = \sum_{i=1}^n (\beta^T x_i + \beta_0 - y_i)\, x_{ij} \quad j=1,\ldots,p$$

---

## 7. Normal Equation (Analytical Solution)

Define augmented matrix and vectors:
$$\hat{X} = \begin{bmatrix}1 & x_1^T \\ 1 & x_2^T \\ \vdots & \vdots \\ 1 & x_n^T\end{bmatrix} \in \mathbb{R}^{n\times(p+1)}, \quad \hat{\beta} = \begin{bmatrix}\beta_0 \\ \beta_1 \\ \vdots \\ \beta_p\end{bmatrix}$$

**Normal equation:**
$$\hat{X}^T \hat{X}\,\hat{\beta} = \hat{X}^T Y$$

**Solution (if $\hat{X}^T\hat{X}$ invertible):**
$$\hat{\beta} = (\hat{X}^T \hat{X})^{-1} \hat{X}^T Y$$

| Case | Condition | Solutions |
|------|-----------|-----------|
| Over-determined | $n \gg p$, $\hat{X}^T\hat{X}$ invertible | Unique |
| Under-determined | $n < p$, $\hat{X}^T\hat{X}$ not invertible | Infinite |

---

## 8. Feature Scaling (Standardisation)

$$X_{\cdot j} \leftarrow \frac{X_{\cdot j} - \text{mean}(X_{\cdot j})}{\text{std}(X_{\cdot j})}, \qquad Y \leftarrow \frac{Y - \text{mean}(Y)}{\text{std}(Y)}$$

---

## 9. Steepest Descent vs. Normal Equation

| | Steepest Descent | Normal Equation |
|-|-----------------|-----------------|
| Type | Iterative | Analytical |
| Step length | Must be chosen | Not needed |
| Large $p$ | Works well | Slow (matrix inversion) |
| **When to use** | $p > 5000$ | $p \leq 5000$ |
