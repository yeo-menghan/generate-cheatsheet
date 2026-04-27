# Lecture 6: Block Coordinate Descent (BCD)

---

## Target Problem

$$\min_{x \in \mathbb{R}^n} F(x) = f(x) + \sum_{i=1}^{n} r_i(x_i)$$

- $f$: convex, differentiable (smooth part)
- $r_i(x_i)$: closed proper convex, **separable** (non-smooth part)

---

## Coordinate-wise Minimizer

**Definition:** $\bar{x}$ is a coordinate-wise minimizer if $\bar{x} \in \text{dom}f$ and:

$$f(\bar{x} + d e_i) \geq f(\bar{x}) \quad \forall\, i \in [n],\; d \in \mathbb{R}$$

Equivalently, for each $i$:

$$\bar{x}_i \in \arg\min_{x_i} f(\bar{x}_1, \ldots, \bar{x}_{i-1}, x_i, \bar{x}_{i+1}, \ldots, \bar{x}_n)$$

### When does coordinate-wise min ⟹ global min?

| Condition | Result |
|---|---|
| $f$ differentiable at $\bar{x}$ | ✅ coord-wise min ⟹ global min (since $\nabla_i f(\bar{x})=0\ \forall i \Rightarrow \nabla f(\bar{x})=0$) |
| $f$ non-differentiable, non-separable | ❌ not guaranteed |
| $f$ non-differentiable, **separable** $r_i(x_i)$ | ✅ coord-wise min ⟹ global min |

---

## Coordinate Descent Algorithm

Cycle through $i = 1, 2, \ldots, n$ and solve one 1D subproblem at a time:

$$x_i^{(k+1)} \leftarrow \arg\min_{x_i}\ f(x_1^{(k+1)}, \ldots, x_{i-1}^{(k+1)},\; x_i,\; x_{i+1}^{(k)}, \ldots, x_n^{(k)}) + r_i(x_i)$$

> Use updated values $x_1^{(k+1)}, \ldots, x_{i-1}^{(k+1)}$ immediately. **Cannot be parallelised.**

---

## Key Tools for Derivations

**Optimality condition** (convex $f$):
$$x^* \in \arg\min f(x) \iff 0 \in \partial f(x^*)$$

**Proximal mapping:**
$$P_{\alpha g}(x) = \arg\min_y \left\{ g(y) + \tfrac{1}{2}\|y - x\|^2 \right\}$$

**Characterisation:**
$$y = P_g(x) \iff x \in y + \partial g(y)$$

**Special cases:**
$$P_{\lambda|\cdot|}(x) = S_\lambda(x) \qquad P_{\delta_C}(x) = \Pi_C(x)$$

**Soft-thresholding operator:**
$$S_\lambda(t) = \text{sign}(t)\cdot\max\{|t| - \lambda,\; 0\}$$

**Subdifferential of $|\cdot|$ at 0:**
$$\partial|0| = [-1, 1]$$

---

## Application 1: Linear Regression

$$\min_{\beta \in \mathbb{R}^p} \tfrac{1}{2}\|X\beta - Y\|^2$$

**Update rule:**
$$\beta_i \leftarrow \frac{X_{\cdot i}^T (Y - X_{-i}\beta_{-i})}{\|X_{\cdot i}\|^2} = \beta_i - \frac{X_{\cdot i}^T(X\beta - Y)}{\|X_{\cdot i}\|^2}$$

> Parameter-free (no step size). $X_{-i}$ = $X$ with $i$-th column removed; $\beta_{-i}$ = $\beta$ with $i$-th entry removed.

---

## Application 2: Lasso

$$\min_{\beta \in \mathbb{R}^p} \tfrac{1}{2}\|X\beta - Y\|^2 + \lambda\|\beta\|_1, \quad \lambda > 0$$

Non-differentiable term is separable: $\lambda\|\beta\|_1 = \sum_{i=1}^p \lambda|\beta_i|$

**Update rule:**
$$\beta_i \leftarrow S_{\lambda / \|X_{\cdot i}\|^2}\!\left(\frac{X_{\cdot i}^T(Y - X_{-i}\beta_{-i})}{\|X_{\cdot i}\|^2}\right)$$

> Same inner quantity as linear regression, wrapped in soft-thresholding. Produces sparse solutions.

---

## Application 3: Box-Constrained Regression

$$\min_{\beta \in \mathbb{R}^p} \tfrac{1}{2}\|X\beta - Y\|^2 \quad \text{s.t.}\ l \leq \beta \leq u$$

Indicator $\delta_C(\beta) = \sum_i \delta_{C_i}(\beta_i)$ is separable.

**Update rule:**
$$\beta_i \leftarrow \Pi_{C_i}\!\left(\frac{X_{\cdot i}^T(Y - X_{-i}\beta_{-i})}{\|X_{\cdot i}\|^2}\right)$$

**Clipping (projection onto $[l_i, u_i]$):**
$$\Pi_{C_i}(t) = \begin{cases} u_i & t > u_i \\ t & l_i \leq t \leq u_i \\ l_i & t < l_i \end{cases}$$

---

## Application 4: SVM — SMO

BCD in **blocks of 2**. Select pair $(\alpha_i, \alpha_j)$ violating KKT, then solve:

$$\min_{\alpha_1, \alpha_2}\ \tfrac{1}{2}K_{11}\alpha_1^2 + \tfrac{1}{2}K_{22}\alpha_2^2 + y_1 y_2 K_{12}\alpha_1\alpha_2 - \alpha_1 - \alpha_2$$
$$\text{s.t.}\quad y_1\alpha_1 + y_2\alpha_2 = \zeta,\quad 0 \leq \alpha_1 \leq C,\quad 0 \leq \alpha_2 \leq C$$

> Reduces to a 1D quadratic over an interval → closed-form solution. Pairs are required to maintain $\sum_i \alpha_i y_i = 0$.

**KKT (complementary slackness) conditions:**

| $\alpha_i$ | Margin condition |
|---|---|
| $\alpha_i = 0$ | $y_i(\beta^T x_i + \beta_0) \geq 1$ |
| $0 < \alpha_i < C$ | $y_i(\beta^T x_i + \beta_0) = 1$ |
| $\alpha_i = C$ | $y_i(\beta^T x_i + \beta_0) \leq 1$ |

---

## Application 5: Parallel Projections

Find $x \in C_1 \cap C_2 \cap \cdots \cap C_m$ via:

$$\min_{x,\, y_i}\ \tfrac{1}{2}\sum_{i=1}^m \|y_i - x\|^2 \quad \text{s.t.}\ y_i \in C_i$$

**BCD updates (repeats until convergence):**
$$y_i \leftarrow \Pi_{C_i}(x) \quad \text{for } i = 1, \ldots, m \qquad \text{(parallelisable)}$$
$$x \leftarrow \frac{1}{m}\sum_{i=1}^m y_i$$

---

## Summary: CD vs Gradient Descent

| | Coordinate Descent | Gradient Descent |
|---|---|---|
| Step size | Not needed | Required ($\alpha$) |
| Per-iteration cost | 1D subproblem | Full gradient $\nabla f$ |
| Parallelisable | ❌ No | ✅ Yes |
| Typical convergence | Fewer iterations | More iterations (fixed $\alpha$) |
