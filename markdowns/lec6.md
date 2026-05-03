# Lecture 6: Block Coordinate Descent (BCD)

## Target Problem

$$\min_{x \in \mathbb{R}^n} F(x) = f(x) + \sum_{i=1}^{n} r_i(x_i)$$

- $f$: convex, differentiable (smooth part)
- $r_i(x_i)$: closed proper convex, **separable** (non-smooth part)

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

## Coordinate Descent Algorithm

Cycle through $i = 1, 2, \ldots, n$ and solve one 1D subproblem at a time:

$$x_i^{(k+1)} \leftarrow \arg\min_{x_i}\ f(x_1^{(k+1)}, \ldots, x_{i-1}^{(k+1)},\; x_i,\; x_{i+1}^{(k)}, \ldots, x_n^{(k)}) + r_i(x_i)$$

> Use updated values $x_1^{(k+1)}, \ldots, x_{i-1}^{(k+1)}$ immediately. **Cannot be parallelised.**

**Example** — $f(x_1, x_2) = x_1^2 + x_1 x_2 + x_2^2$, start at $(x_1, x_2) = (2, 2)$:

- Update $x_1$: $\min_{x_1} x_1^2 + 2x_1 \Rightarrow \frac{\partial}{\partial x_1} = 2x_1 + x_2 = 0 \Rightarrow x_1^{(1)} = -x_2/2 = -1$
- Update $x_2$: $\min_{x_2} (-1)(-1) + (-1)x_2 + x_2^2 \Rightarrow \frac{\partial}{\partial x_2} = x_1 + 2x_2 = 0 \Rightarrow x_2^{(1)} = 1/2$

After 1 step: $(x_1, x_2) = (-1,\ 0.5)$, moving toward $(0, 0)$.

## Key Tools for Derivations

**Optimality condition** (convex $f$): $x^* \in \arg\min f(x) \iff 0 \in \partial f(x^*)$

**Proximal characterisation:** $y = P_g(x) \iff x \in y + \partial g(y)$

**Special cases:** $P_{\lambda|\cdot|}(x) = S_\lambda(x)$, $\quad P_{\delta_C}(x) = \Pi_C(x)$, $\quad \partial|0| = [-1,1]$

## Applications 1–3: Regression Variants

Let $\tilde{\beta}_i = \dfrac{X_{\cdot i}^T(Y - X_{-i}\beta_{-i})}{\|X_{\cdot i}\|^2}$ (OLS update for coordinate $i$). All three share this inner quantity:

| Problem | Update rule |
|---|---|
| Linear regression | $\beta_i \leftarrow \tilde{\beta}_i$ |
| Lasso ($+\lambda\|\beta\|_1$) | $\beta_i \leftarrow S_{\lambda/\|X_{\cdot i}\|^2}(\tilde{\beta}_i)$ |
| Box-constrained ($l \leq \beta \leq u$) | $\beta_i \leftarrow \Pi_{[l_i, u_i]}(\tilde{\beta}_i)$ |

> $X_{-i}$ = $X$ with $i$-th column removed; $\Pi_{[l,u]}(t) = \min(u, \max(l, t))$. No step size needed.

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

## Application 5: Parallel Projections

Find $x \in C_1 \cap C_2 \cap \cdots \cap C_m$ via:

$$\min_{x,\, y_i}\ \tfrac{1}{2}\sum_{i=1}^m \|y_i - x\|^2 \quad \text{s.t.}\ y_i \in C_i$$

**BCD updates (repeats until convergence):**
$$y_i \leftarrow \Pi_{C_i}(x) \quad \text{for } i = 1, \ldots, m \qquad \text{(parallelisable)}$$
$$x \leftarrow \frac{1}{m}\sum_{i=1}^m y_i$$

## Summary: CD vs Gradient Descent

| | Coordinate Descent | Gradient Descent |
|---|---|---|
| Step size | Not needed | Required ($\alpha$) |
| Per-iteration cost | 1D subproblem | Full gradient $\nabla f$ |
| Parallelisable | ❌ No | ✅ Yes |
| Typical convergence | Fewer iterations | More iterations (fixed $\alpha$) |
