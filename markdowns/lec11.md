# Lecture 11: Second Order Methods

## 1. Rates of Convergence

| Type | Condition | Example |
|------|-----------|---------|
| **Q-linear** | $\frac{\|x^{(k+1)}-x^*\|}{\|x^{(k)}-x^*\|} \leq r \in (0,1)$ | $1 + 0.8^k$ |
| **Q-superlinear** | $\lim_{k\to\infty}\frac{\|x^{(k+1)}-x^*\|}{\|x^{(k)}-x^*\|} = 0$ | $1 + k^{-k}$ |
| **Q-quadratic** | $\frac{\|x^{(k+1)}-x^*\|}{\|x^{(k)}-x^*\|^2} \leq M$ for some $M>0$ | $1 + (0.8)^{2^k}$ |

**Hierarchy:** Q-quadratic > Q-superlinear > Q-linear (eventually faster)

**Key facts:**
- Q-quadratic ⟹ Q-superlinear ⟹ Q-linear
- $1/k$ is **not** Q-linearly convergent; $1/k!$ is Q-superlinear but **not** Q-quadratic

## 2. General Framework

For $\min_x f(x)$, differentiable:
$$x^{(k+1)} = x^{(k)} - \alpha_k P_k \nabla f(x^{(k)})$$

| $P_k$ | Method |
|-------|--------|
| $I$ | Gradient (steepest) descent |
| $[H_f(x^{(k)})]^{-1}$ | Pure Newton's method |
| $[H_f(x^{(k)}) + \tau_k I]^{-1}$ | Modified Newton |
| $H_k \approx [H_f(x^{(k)})]^{-1}$ | Quasi-Newton (BFGS) |

## 3. Pure Newton's Method

**Derivation:** Minimize 2nd-order Taylor approximation of $f$.

**Newton direction:**
$$p^{(k)} = -[H_f(x^{(k)})]^{-1} \nabla f(x^{(k)})$$
equivalently, solve the **linear system**:
$$H_f(x^{(k)})\, p = -\nabla f(x^{(k)})$$

**Update:** $x^{(k+1)} = x^{(k)} + p^{(k)}$ (step size = 1)

**Descent direction condition:** $p^{(k)}$ is a descent direction iff $H_f(x^{(k)}) \succ 0$:
$$\nabla f(x^{(k)})^T p^{(k)} = -(p^{(k)})^T H_f(x^{(k)}) p^{(k)} < 0$$

**Convergence:** Q-quadratic (locally, near $x^*$), given:
1. $f$ twice differentiable
2. $H_f(\cdot)$ locally Lipschitz at $x^*$: $\|H_f(x^*) - H_f(x)\| \leq L\|x^*-x\|$
3. Sufficient conditions (SOSC) at $x^*$

**Limitations:** May diverge from remote starting points; requires Hessian ($O(n^3)$ to solve).

## 4. Practical Newton's Method

### 4a. Line Search Newton with Modification

$$x^{(k+1)} = x^{(k)} + \alpha_k p^{(k)}, \quad p^{(k)} = -[H_f(x^{(k)}) + \tau_k I]^{-1}\nabla f(x^{(k)})$$

**Hessian modification:** If $H_f(x^{(k)}) \not\succ 0$, choose $\tau_k > -\lambda_{\min}(H_f(x^{(k)}))$ so $B_k = H_f(x^{(k)}) + \tau_k I \succ 0$.

- If $\tau_k$ too large: $p^{(k)} \approx -\tau_k^{-1}\nabla f(x^{(k)})$ (degenerates to slow gradient descent)

**Line search (Armijo backtracking):**
$$f(x^{(k)} + \alpha p^{(k)}) \leq f(x^{(k)}) + c_1 \alpha \nabla f(x^{(k)})^T p^{(k)}$$
Start with $\alpha=1$, reduce by $\alpha \leftarrow \rho\alpha$ until satisfied.

**Convergence rate:**
- If $H_f(x^*) \succ 0$: $\tau_k \to 0$, reduces to pure Newton → **Q-quadratic**
- Otherwise: may be only **linear**

### 4b. BFGS (Quasi-Newton)

Approximates $H_k \approx [H_f(x^{(k)})]^{-1}$ without computing the true Hessian.

**Secant equation** (the key constraint on $B_{k+1} \approx H_f$):
$$B_{k+1} s_k = y_k$$
where $s_k = x^{(k+1)} - x^{(k)}$, $\quad y_k = \nabla f(x^{(k+1)}) - \nabla f(x^{(k)})$

**Curvature condition** (required for $B_{k+1} \succ 0$): $y_k^T s_k > 0$

**BFGS update formula** (updates inverse Hessian approximation $H_k$):
$$H_{k+1} = (I - \rho_k s_k y_k^T) H_k (I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T, \quad \rho_k = \frac{1}{y_k^T s_k}$$

**Equivalent $B_k$ update (rank-2):**
$$B_{k+1} = B_k - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k} + \frac{y_k y_k^T}{y_k^T s_k}$$

**Per-iteration cost:** $O(n^2)$. **Convergence:** Q-superlinear.

## 5. Trust-Region Method

**Model function:**
$$m_k(p) = f(x^{(k)}) + \nabla f(x^{(k)})^T p + \frac{1}{2} p^T B_k p$$

**Subproblem:** $\min_p\ m_k(p) \quad \text{s.t.} \quad \|p\| \leq \Delta_k$

**Ratio for adapting radius:**
$$\rho_k = \frac{f(x^{(k)}) - f(x^{(k)}+p^{(k)})}{m_k(0) - m_k(p^{(k)})} = \frac{\text{actual reduction}}{\text{predicted reduction}}$$

| $\rho_k$ | Action |
|----------|--------|
| $< 1/4$ | Shrink: $\Delta_{k+1} = \frac{1}{4}\Delta_k$, reject step |
| $> 3/4$ and $\|p^{(k)}\| = \Delta_k$ | Expand: $\Delta_{k+1} = \min(2\Delta_k, \hat\Delta)$ |
| otherwise | Keep: $\Delta_{k+1} = \Delta_k$ |
| $> \eta$ | Accept step: $x^{(k+1)} = x^{(k)} + p^{(k)}$ |

**Full step** (when $B_k \succ 0$ and $\|B_k^{-1}\nabla f(x^{(k)})\| \leq \Delta_k$):
$$p^{(k)} = -B_k^{-1}\nabla f(x^{(k)})$$

## 6. Proximal Newton (Non-smooth $g$)

For $\min_x f(x) + g(x)$ ($f$ smooth, $g$ non-smooth):
$$x^{(k+1)} = \arg\min_x \left\{ \nabla f(x^{(k)})^T(x-x^{(k)}) + \frac{1}{2}(x-x^{(k)})^T B_k (x-x^{(k)}) + g(x) \right\}$$

---

## Quick Reference: Convergence Rates

| Method | Rate |
|--------|------|
| Gradient descent | Q-linear |
| Newton's (pure, local) | Q-quadratic |
| Newton's (modified + line search, at $x^*$ with $H\succ0$) | Q-quadratic |
| Newton's (at $x^*$ with $H\not\succ0$) | Q-linear |
| BFGS | Q-superlinear |
| Trust-region Newton (at SOSC point) | Q-superlinear |
