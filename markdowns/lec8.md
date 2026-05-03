# Lecture 8: Recommendation Systems & Matrix Completion

## 1. Setup

- **Utility matrix** $R \in \mathbb{R}^{m \times n}$: $R_{ij}$ = rating of user $i$ on item $j$
- **$\Omega$**: index set of known entries
- **Goal**: predict missing entries

## 2. Collaborative Filtering

**Similarity measures** (user $x$ vs user $y$):

| Measure | Formula | Range |
|---|---|---|
| Jaccard | $\dfrac{\lvert S_x \cap S_y\rvert}{\lvert S_x \cup S_y\rvert}$ | $[0,1]$ |
| Cosine | $\dfrac{r_x^T r_y}{\lVert r_x\rVert \lVert r_y\rVert}$ | $[0,1]$ |
| Normalized cosine | $\dfrac{(r_x-\bar{r}_x)^T(r_y-\bar{r}_y)}{\lVert r_x-\bar{r}_x\rVert\lVert r_y-\bar{r}_y\rVert}$ | $[-1,1]$ |

**Prediction** for user $x$, item $i$ (let $N(x,i)$ = neighbors who rated $i$):
$$r_{xi} = \frac{\sum_{y \in N(x,i)} \text{Sim}(x,y)\cdot r_{yi}}{\sum_{y \in N(x,i)} \text{Sim}(x,y)}$$

**Item-Item CF**: same formula but swap user/item roles — find similar items $j \in N(i,x)$.

## 3. Latent Factor Model

Assume low-rank structure $R \approx WH$, where $W \in \mathbb{R}^{m \times k}$, $H \in \mathbb{R}^{k \times n}$, $k \ll m,n$.

$$\min_{W,H}\ \frac{1}{2}\sum_{(i,j)\in\Omega}(R_{ij} - W_{i\cdot}H_{\cdot j})^2 + \frac{\lambda}{2}\left(\lVert W\rVert_F^2 + \lVert H\rVert_F^2\right)$$

**Gradients:**
$$[\nabla_W F]_{it} = -\sum_{j:(i,j)\in\Omega}(R_{ij} - W_{i\cdot}H_{\cdot j})H_{tj} + \lambda W_{it}$$
$$[\nabla_H F]_{tj} = -\sum_{i:(i,j)\in\Omega}(R_{ij} - W_{i\cdot}H_{\cdot j})W_{it} + \lambda H_{tj}$$

> Non-convex in $(W,H)$ jointly; use gradient descent or SGD (sample one rating per step).

## 4. Matrix Completion & Nuclear Norm

**Rank minimization** (NP-hard) → relax to **nuclear norm** (convex envelope of rank):

$$\min_X \text{rank}(X) \quad \xrightarrow{\text{relax}} \quad \min_X \lVert X\rVert_* \quad \text{s.t.}\ X_{ij} = M_{ij},\ (i,j)\in\Omega$$

$$\lVert X\rVert_* = \sum_i \sigma_i(X) \quad \text{(sum of singular values, analogous to } \ell_1 \text{ on singular values)}$$

**SVD:** $X = U\Sigma V^T = \sum_i \sigma_i u_i v_i^T$

**Recovery guarantee**: if $\text{rank}(X^*) = k$ and $|\Omega| \geq C \cdot n \cdot k \cdot \log^2(n)$ (uniform random), nuclear norm minimization recovers $X^*$ w.h.p.

## 5. Algorithms

### Algorithm 1: SDP
$$\lVert X\rVert_* = \min_{W_1,W_2} \tfrac{1}{2}(\text{Tr}(W_1)+\text{Tr}(W_2)) \quad \text{s.t.}\ \begin{bmatrix}W_1 & X \\ X^T & W_2\end{bmatrix} \succeq 0$$

Plug in $X_{ij} = M_{ij}$ constraints → standard SDP, solve with CVX/Mosek.

### Algorithm 2: Proximal Gradient

$$\min_X\ \underbrace{\frac{1}{2}\sum_{(i,j)\in\Omega}(X_{ij}-M_{ij})^2}_{f(X)} + \underbrace{\mu\lVert X\rVert_*}_{g(X)}$$

**Proximal map of $\mu\lVert\cdot\rVert_*$** — soft-threshold singular values:
1. SVD: $Y = U\,\text{diag}(\sigma)\,V^T$
2. $\gamma_i = \max(\sigma_i - \mu,\ 0)$
3. $\text{prox}_{\mu\lVert\cdot\rVert_*}(Y) = U\,\text{diag}(\gamma)\,V^T$

**PG iteration:** $X^{(t+1)} \leftarrow \text{prox}_{\mu\lVert\cdot\rVert_*}\!\left(X^{(t)} - \alpha\nabla f(X^{(t)})\right)$

## 6. Method Comparison

| Method | Pros | Cons |
|---|---|---|
| User/Item CF | Simple, interpretable | Doesn't scale; cold-start problem |
| Latent Factor | Scalable | Non-convex, needs tuning |
| Nuclear Norm (SDP/PG) | Convex, theoretical guarantees | Expensive (SVD per iteration) |
