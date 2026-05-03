# Lecture 4: Proximal Gradient Method

## 1. Norms

| Norm | Definition |
|------|-----------|
| $\ell_1$ | $\lVert x\rVert_1 = \sum_i \lvert x_i\rvert$ |
| $\ell_2$ | $\lVert x\rVert_2 = \sqrt{\sum_i x_i^2}$ |
| $\ell_\infty$ | $\lVert x\rVert_\infty = \max_i \lvert x_i\rvert$ |
| $\ell_p$ | $\lVert x\rVert_p = \left(\sum_i \lvert x_i\rvert^p\right)^{1/p}$ |
| Frobenius | $\lVert A\rVert_F^2 = \langle A, A \rangle = \text{Tr}(A^T A)$ |

**Inner product (matrices):** $\langle A, B \rangle = \text{Tr}(A^T B) = \sum_i \sum_j A_{ij} B_{ij}$

**Inner product (vectors):** $\langle x, y \rangle = x^T y = \sum_i x_i y_i$

## 2. Projection onto Closed Convex Set $C$

$$\Pi_C(z) = \arg\min_{x \in C} \frac{1}{2}\|x - z\|^2$$

**Characterisation:** $x^* = \Pi_C(z) \iff \langle z - x^*, x - x^* \rangle \leq 0 \quad \forall x \in C$

| Set $C$ | $\Pi_C(z)$ |
|---------|------------|
| $\mathbb{R}^n_+$ (positive orthant) | $\max\{z, 0\}$ (elementwise) |
| $\ell_2$-ball $\{\|x\|_2 \leq 1\}$ | $z / \max\{\|z\|_2, 1\}$ |
| $\mathbb{S}^n_+$ (PSD cone), $A = Q\Lambda Q^T$ | $Q \cdot \text{diag}(\max\{\lambda_i, 0\}) \cdot Q^T$ |

## 3. Normal Cone

$$\mathcal{N}_C(\bar{x}) = \{ z \mid \langle z, x - \bar{x} \rangle \leq 0 \quad \forall x \in C \}$$

**Key equivalence:** $u \in \mathcal{N}_C(y) \iff y = \Pi_C(y + u)$

**Property:** If $\bar{x} \in \text{int}(C)$, then $\mathcal{N}_C(\bar{x}) = \{0\}$

**Example — $C = [0, 1]$:**

| $\bar{x}$ | $\mathcal{N}_C(\bar{x})$ |
|-----------|--------------------------|
| $0$ | $(-\infty, 0]$ |
| $1$ | $[0, +\infty)$ |
| $(0, 1)$ | $\{0\}$ |
| $\bar{x} \notin C$ | $\emptyset$ |

**Example — $C = \{ x \in \mathbb{R}^2 : \|x\| \leq 1 \}$:**

| $\bar{x}$ | $\mathcal{N}_C(\bar{x})$ |
|-----------|--------------------------|
| $\|\bar{x}\| = 1$ | $\{ \lambda\bar{x} : \lambda \geq 0 \}$ |
| $\|\bar{x}\| < 1$ | $\{0\}$ |

## 4. Subdifferential

$$\partial f(x) = \{ v \mid f(z) \geq f(x) + \langle v, z - x \rangle \quad \forall z \}$$

- If $f$ differentiable at $x$: $\partial f(x) = \{\nabla f(x)\}$
- **Global optimality:** $\bar{x}$ is a global minimiser $\iff 0 \in \partial f(\bar{x})$
- **Indicator function:** $\partial \delta_C(x) = \mathcal{N}_C(x)$ for $x \in C$

**Example — $f(x) = |x|$:**

| $x$ | $\partial f(x)$ |
|-----|-----------------|
| $x < 0$ | $\{-1\}$ |
| $x = 0$ | $[-1, 1]$ |
| $x > 0$ | $\{1\}$ |

**Lasso sparsity condition** — at optimum $\beta$:

$$\beta_i < 0 \implies [X^T(X\beta - Y)]_i = \lambda$$
$$\beta_i = 0 \iff |[X^T(X\beta - Y)]_i| \leq \lambda \quad \leftarrow \text{sparsity!}$$
$$\beta_i > 0 \implies [X^T(X\beta - Y)]_i = -\lambda$$

## 5. Fenchel Conjugate

$$f^*(y) = \sup_x \{ \langle y, x \rangle - f(x) \}$$

**$f^*$ is always convex and closed** (even if $f$ is not).

If $f$ is closed proper convex, then $(f^*)^* = f$.

**Key triple equivalence:**

$$f(x) + f^*(y) = \langle x, y \rangle \iff y \in \partial f(x) \iff x \in \partial f^*(y)$$

**Examples:**

| $f(x)$ | $f^*(y)$ |
|--------|----------|
| $\|x\|_1$ | $\delta_C(y)$, $C = \{ y : \|y\|_\infty \leq 1 \}$ |
| $\delta_C(x)$ (indicator) | $\sup\{ \langle y, x \rangle : x \in C \}$ (support function) |

## 6. Moreau Envelope & Proximal Operator

$$\text{prox}_f(x) = \arg\min_y \left\{ f(y) + \frac{1}{2}\|y - x\|^2 \right\} \quad \leftarrow \text{proximal mapping}$$

$$M_f(x) = \min_y \left\{ f(y) + \frac{1}{2}\|y - x\|^2 \right\} \quad \leftarrow \text{Moreau envelope}$$

**Properties:**

- $\nabla M_f(x) = x - \text{prox}_f(x)$ &nbsp; ($M_f$ is always differentiable)
- $\arg\min f = \arg\min M_f$
- $\text{prox}_{\delta_C}(x) = \Pi_C(x)$

**Moreau Decomposition:**

$$x = \text{prox}_f(x) + \text{prox}_{f^*}(x)$$
$$\frac{1}{2}\|x\|^2 = M_f(x) + M_{f^*}(x)$$

**Soft Thresholding** — prox of $f(x) = \lambda|x|$:

$$\text{prox}_f(x) = S_\lambda(x) = \text{sign}(x) \cdot \max\{|x| - \lambda,\ 0\}$$

Applied elementwise to $x = [x_1; \ldots; x_n]$:

$$[S_\lambda(x)]_i = \text{sign}(x_i) \cdot \max\{|x_i| - \lambda,\ 0\}$$

**Huber function** (Moreau envelope of $f(x) = \lambda|x|$):

$$M_f(x) = \begin{cases} \frac{1}{2}x^2 & \text{if } |x| \leq \lambda \\ \lambda|x| - \frac{\lambda^2}{2} & \text{if } |x| > \lambda \end{cases}$$

## 7. Proximal Gradient (PG) Method

**Problem:** $\min_\beta\ f(\beta) + g(\beta)$, where $f$ is smooth and $g$ is convex non-smooth.

**Key insight** — gradient step on $f$ only, then prox on $g$:

$$\beta^{(k+1)} = \text{prox}_{\alpha g}\!\left( \beta^{(k)} - \alpha \nabla f(\beta^{(k)}) \right)$$

**Algorithm:**

1. Choose $\beta^{(0)}$, step size $\alpha > 0$
2. Repeat: $\beta^{(k+1)} = \text{prox}_{\alpha g}\!\left( \beta^{(k)} - \alpha \nabla f(\beta^{(k)}) \right)$
3. Until convergence

**Convergence:** $f(\beta^{(k)}) + g(\beta^{(k)}) - \text{optimal} \leq O(1/k)$

## 8. Accelerated Proximal Gradient (APG) Method

**Algorithm (FISTA-style):** Choose $\beta^{(0)}$, step size $\alpha > 0$, $t_0 = t_1 = 1$. Repeat:

$$\bar{\beta}^{(k)} = \beta^{(k)} + \frac{t_k - 1}{t_{k+1}} \left( \beta^{(k)} - \beta^{(k-1)} \right) \quad \leftarrow \text{momentum}$$

$$\beta^{(k+1)} = \text{prox}_{\alpha g}\!\left( \bar{\beta}^{(k)} - \alpha \nabla f(\bar{\beta}^{(k)}) \right)$$

$$t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}$$

**Convergence:** $f(\beta^{(k)}) + g(\beta^{(k)}) - \text{optimal} \leq O(1/k^2)$

**Step size rule:** $\alpha \in (0, 1/L)$, where $L$ = Lipschitz constant of $\nabla f$

## 9. APG Applied to Lasso

$$\min_\beta\ \frac{1}{2}\|X\beta - Y\|^2 + \lambda\|\beta\|_1$$

- $\nabla f(\beta) = X^T(X\beta - Y)$
- Lipschitz constant: $L = \lambda_{\max}(X^T X)$
- Step size: $\alpha = 1/L$

**Iteration:**

$$\bar{\beta}^{(k)} = \beta^{(k)} + \frac{t_k - 1}{t_{k+1}} \left( \beta^{(k)} - \beta^{(k-1)} \right)$$

$$\beta^{(k+1)} = S_{\lambda/L}\!\left( \bar{\beta}^{(k)} - \frac{1}{L} X^T(X\bar{\beta}^{(k)} - Y) \right)$$

**Optimality condition:** $\beta^*$ is optimal $\iff \beta^* = \text{prox}_g(\beta^* - \nabla f(\beta^*))$

**Stopping criterion** (tolerance $\varepsilon > 0$):

$$\left\| \beta^{(k)} - S_\lambda\!\left( \beta^{(k)} - X^T(X\beta^{(k)} - Y) \right) \right\| < \varepsilon$$

## 10. Complexity Summary

| Method | Convergence rate | Iterations to $10^{-4}$ error |
|--------|-----------------|-------------------------------|
| PG | $O(1/k)$ | $\sim O(10^4)$ |
| APG | $O(1/k^2)$ | $\sim O(10^2)$ |

- APG has the **same per-iteration cost** as PG (one prox + one gradient eval)
- **Restart trick:** rerun APG every 100–200 iterations from the latest iterate
