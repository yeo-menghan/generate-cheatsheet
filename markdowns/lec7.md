# Lecture 7: NMF

## Core Problem
$$\min_{W,H} \frac{1}{2}\|V - WH\|_F^2 \quad \text{s.t.} \quad W \geq 0,\ H \geq 0$$

| Matrix | Dimensions | Role |
|--------|-----------|------|
| V | m × n | Data (columns = data points) |
| W | m × r | Basis vectors |
| H | r × n | Coordinates/coefficients |

**Bi-convex** (non-convex jointly, but convex in W or H separately).

## Variants

| Variant | Extra Constraint | Use Case |
|---------|-----------------|----------|
| Standard NMF | — | General |
| Orthogonal NMF | $HH^T = I_r$ | Hard clustering |
| Symmetric NMF | $H = W^T$, so $V \approx WW^T$ | Community detection |
| Sparse NMF | $+ \lambda_W\|W\|_1 + \lambda_H\|H\|_1$ | Localized features |

**Key lemma (Orthogonal NMF):** $H \geq 0$ and $HH^T = I_r$ ⟹ each column of $H$ has **at most one positive entry**.

## Algorithms

### BCD / HALS
Update columns of W, then rows of H cyclically:

$$W_{\cdot i} \leftarrow \Pi_+\!\left(\frac{(V - W_{\cdot(-i)}H_{(-i)\cdot})\,H_{i\cdot}^T}{\|H_{i\cdot}\|^2}\right), \qquad H_{i\cdot} \leftarrow \Pi_+\!\left(\frac{W_{\cdot i}^T(V - W_{\cdot(-i)}H_{(-i)\cdot})}{\|W_{\cdot i}\|^2}\right)$$

### Proximal Gradient (PG)
$$W^{(k+1)} = \Pi_+\!\left(W^{(k)} + \alpha_k R^{(k)}(H^{(k)})^T\right), \qquad H^{(k+1)} = \Pi_+\!\left(H^{(k)} + \alpha_k (W^{(k)})^T R^{(k)}\right)$$
where $R^{(k)} = V - W^{(k)}H^{(k)}$.

### Multiplicative Update (MU)
$$W \leftarrow W \odot \frac{VH^T}{WHH^T}, \qquad H \leftarrow H \odot \frac{W^TV}{W^TWH}$$

### ALS (default in sklearn)
Solve unconstrained least squares for H, then W; project onto $\mathbb{R}^+$ after each step.

## Key Equations

| Item | Formula |
|------|---------|
| Frobenius norm | $\|A\|_F^2 = \text{Tr}(A^TA) = \sum_{i,j}A_{ij}^2$ |
| $\ell_1$ norm | $\|A\|_1 = \sum_{i,j}\|A_{ij}\|$ |
| Gradient w.r.t. W | $\nabla_W f = -(V-WH)H^T$ |
| Gradient w.r.t. H | $\nabla_H f = -W^T(V-WH)$ |
| Projection | $\Pi_+(x) = \max(x,\, 0)$ element-wise |
| Rank inequality | $\text{rank}(V) \leq \text{rank}_+(V) \leq \min(m,n)$ |

## Cluster Membership Rules

- **Orthogonal / Standard NMF:** point $j \in$ cluster $k$ if $H_{kj} > H_{ij}\ \forall i \neq k$
- **Symmetric NMF:** node $j \in$ community $k$ if $W_{jk} > W_{ji}\ \forall i \neq k$
