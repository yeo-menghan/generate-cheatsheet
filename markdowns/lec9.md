# Lecture 9: PCA, Robust PCA & ADMM

## 1. PCA

**Goal:** Find direction $w$ with $\|w\| = 1$ of greatest variance in data $X \in \mathbb{R}^{n \times p}$.

**Covariance matrix:**

$$\Sigma = \frac{1}{n} \sum_{i=1}^{n} x_i x_i^T = \frac{1}{n} X^T X \quad \text{(assuming zero-mean data)}$$

**Optimization problem:**

$$\max_{w} \; w^T X^T X w \quad \text{s.t.} \quad \|w\| = 1$$

Equivalent to maximizing the **Rayleigh quotient**:

$$\max_{w} \; \frac{w^T X^T X w}{w^T w}$$

**Solution:** $w$ = eigenvector of $X^T X$ with the **largest eigenvalue** $\lambda_1$.

**Steps to find the principal subspace:**
1. Compute $\Sigma = \frac{1}{n} \sum_{i=1}^{n} x_i x_i^T$
2. Find eigenvalues via $\det(\Sigma - \lambda I) = 0$
3. Find eigenvector $w$ for the largest $\lambda$ via $(\Sigma - \lambda I)w = 0$

**$k$-th component:** eigenvector of $X^T X$ with the $k$-th largest eigenvalue.

**Equivalent low-rank formulation:**

$$\min_{L:\,\text{rank}(L)=r} \|X - L\|_F$$

> ⚠️ PCA is **sensitive to outliers** — a few corrupted entries can distort the principal components.

## 2. Robust PCA

**Model:** $X = L_0 + S_0$, where $L_0$ is low-rank and $S_0$ is sparse.

**Non-convex (NP-hard) formulation:**

$$\min_{L,\,S} \; \text{rank}(L) + \lambda \|S\|_0 \quad \text{s.t.} \quad L + S = X$$

**Convex relaxation:**

$$\min_{L,\,S} \; \|L\|_* + \lambda \|S\|_1 \quad \text{s.t.} \quad L + S = X$$

| Term | Meaning | Relaxes |
|------|---------|---------|
| $\|L\|_* = \text{tr}(\sqrt{L^T L})$ | Nuclear norm = sum of singular values | $\text{rank}(L)$ |
| $\|S\|_1$ | Sum of absolute values of all entries | $\|S\|_0$ |

**Recommended $\lambda$:**

$$\lambda = \frac{1}{\sqrt{\max(m,\, n)}}$$

## 3. ADMM

**Target problem (2-block separable):**

$$\min_{y,\,z} \; f(y) + g(z) \quad \text{s.t.} \quad Ay + Bz = c$$

**Augmented Lagrangian** ($\sigma > 0$):

$$\mathcal{L}_\sigma(y, z, x) = f(y) + g(z) + \langle x,\, Ay + Bz - c \rangle + \frac{\sigma}{2}\|Ay + Bz - c\|^2$$

Using completing the square, equivalently:

$$\mathcal{L}_\sigma(y, z, x) = f(y) + g(z) + \frac{\sigma}{2}\left\|Ay + Bz - c + \sigma^{-1}x\right\|^2 - \frac{1}{2\sigma}\|x\|^2$$

**ADMM iterations:**

$$y^{(k+1)} \leftarrow \arg\min_{y} \; \mathcal{L}_\sigma\!\left(y,\, z^{(k)},\, x^{(k)}\right)$$

$$z^{(k+1)} \leftarrow \arg\min_{z} \; \mathcal{L}_\sigma\!\left(y^{(k+1)},\, z,\, x^{(k)}\right)$$

$$x^{(k+1)} \leftarrow x^{(k)} + \tau\sigma\!\left(Ay^{(k+1)} + Bz^{(k+1)} - c\right)$$

**Parameters:** $\sigma > 0$, $\quad 0 < \tau < \dfrac{1 + \sqrt{5}}{2} \approx 1.618$

**Key completing-the-square identity:**

$$\langle X, Y \rangle + \frac{\sigma}{2}\|Y\|^2 = \frac{\sigma}{2}\left\|Y + \sigma^{-1}X\right\|^2 - \frac{1}{2\sigma}\|X\|^2$$

**Proximal operator:**

$$\mathbf{P}_{\frac{1}{\sigma}f}(v) = \arg\min_{y} \left\{ \frac{1}{\sigma} f(y) + \frac{1}{2}\|y - v\|^2 \right\}$$

## 4. ADMM for Robust PCA

**Problem:** $\min_{L,S}\; \|L\|_* + \lambda\|S\|_1 \;\text{ s.t. }\; L + S = X \quad \Rightarrow \quad A = I,\; B = I,\; c = X$

**Augmented Lagrangian:**

$$\mathcal{L}_\sigma(L, S, Z) = \|L\|_* + \lambda\|S\|_1 + \frac{\sigma}{2}\left\|L + S - X + \sigma^{-1}Z\right\|_F^2 - \frac{1}{2\sigma}\|Z\|_F^2$$

**Subproblem-$L$** (singular value soft-thresholding):

1. Compute SVD: $\;T^{(k)} = X - S^{(k)} - \sigma^{-1}Z^{(k)} = U^{(k)}\,\text{Diag}(d^{(k)})\,(V^{(k)})^T$
2. Soft-threshold singular values: $\;\gamma^{(k)} = \mathcal{S}_{1/\sigma}(d^{(k)})$
3. Update: $\;L^{(k+1)} = U^{(k)}\,\text{Diag}(\gamma^{(k)})\,(V^{(k)})^T$

**Subproblem-$S$** (element-wise soft-thresholding):

$$S^{(k+1)} = \mathcal{S}_{\lambda/\sigma}\!\left(X - L^{(k+1)} - \sigma^{-1}Z^{(k)}\right)$$

**Multiplier update:**

$$Z^{(k+1)} = Z^{(k)} + \tau\sigma\!\left(L^{(k+1)} + S^{(k+1)} - X\right)$$

## 5. ADMM for Lasso

**Lasso:** $\;\min_{\beta \in \mathbb{R}^p}\; \dfrac{1}{2}\|X\beta - Y\|^2 + \lambda\|\beta\|_1, \quad X \in \mathbb{R}^{n \times p},\; Y \in \mathbb{R}^n$

### Primal ADMM — use when $p < n$

Introduce slack $u = \beta$:

$$\min_{\beta,\,u} \; \frac{1}{2}\|X\beta - Y\|^2 + \lambda\|u\|_1 \quad \text{s.t.} \quad \beta - u = 0$$

**Iterations:**

$$\beta^{(k+1)} = \left(\sigma I + X^T X\right)^{-1}\!\left(X^T Y + \sigma u^{(k)} - \xi^{(k)}\right) \quad [p \times p \text{ linear system}]$$

$$u^{(k+1)} = \mathcal{S}_{\lambda/\sigma}\!\left(\beta^{(k+1)} + \sigma^{-1}\xi^{(k)}\right)$$

$$\xi^{(k+1)} = \xi^{(k)} + \tau\sigma\!\left(\beta^{(k+1)} - u^{(k+1)}\right)$$

### Dual ADMM — use when $n < p$

Dual problem:

$$\min_{y,\,v} \; \frac{1}{2}\|y\|^2 - \langle y, Y\rangle + \delta_{B_\lambda}(v) \quad \text{s.t.} \quad X^T y + v = 0$$

**Iterations:**

$$y^{(k+1)} = \left(I + \sigma X X^T\right)^{-1}\!\left(Y - X\!\left(\beta^{(k)} + \sigma v^{(k)}\right)\right) \quad [n \times n \text{ linear system}]$$

$$v^{(k+1)} = \Pi_{B_\lambda}\!\left(-X^T y^{(k+1)} - \sigma^{-1}\beta^{(k)}\right)$$

$$\beta^{(k+1)} = \beta^{(k)} + \tau\sigma\!\left(X^T y^{(k+1)} + v^{(k+1)}\right)$$

where $B_\lambda = \{v \mid \|v\|_\infty \leq \lambda\}$ and $\Pi_{B_\lambda}(s)_i = \text{clamp}(s_i,\,-\lambda,\,\lambda)$.

## 6. Key Operators

| Operator | Formula | Used in |
|----------|---------|---------|
| **Soft-thresholding** $\mathcal{S}_\tau(x)_i$ | $\text{sign}(x_i)\cdot\max(\|x_i\| - \tau,\; 0)$ | $\ell_1$ proximal |
| **SVD soft-threshold** | Apply $\mathcal{S}_{1/\sigma}$ to singular values | Nuclear norm proximal |
| **Projection onto $B_\lambda$** | Clamp each entry to $[-\lambda,\, \lambda]$ | Dual Lasso |
| **Projection onto set $C$** | $\Pi_C(v) = \arg\min_{z \in C}\|z - v\|$ | Constrained problems |

## 7. Practical Notes

- **$\sigma$ too large** $\Rightarrow$ insufficient minimization of $f + g$
- **$\sigma$ too small** $\Rightarrow$ insufficient feasibility enforcement
- Common step lengths: $\tau = 1$ or $\tau = 1.618 \approx \frac{1+\sqrt{5}}{2}$
- Pre-compute and cache $(\sigma I + X^T X)^{-1}$ or $(I + \sigma X X^T)^{-1}$ before iterating
- Choose **primal vs. dual ADMM** based on problem dimension:
  - $p < n \;\Rightarrow\;$ primal ADMM ($p \times p$ system)
  - $n < p \;\Rightarrow\;$ dual ADMM ($n \times n$ system)
