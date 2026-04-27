# Lecture 8: Recommendation Systems & Matrix Completion

## 1. Utility Matrix and Core Concepts

### Definition
- **Utility Matrix R**: m × n matrix where R_ij = rating of user i on item j
- **Ω**: Index set of known entries: Ω = {(i,j) | R_ij is known}
- **Sparsity**: Most entries are unknown (blanks)

### Goal
Predict missing ratings (blank entries) in the utility matrix

## 2. Collaborative Filtering (CF)

### A. User-User CF

**Step 1: Find Similar Users**
For user x, identify neighborhood N_x using similarity metrics:

#### Similarity Measures:

**Jaccard Similarity** (for binary matrices)
$$\text{Sim}(x,y) = \frac{|S_x \cap S_y|}{|S_x \cup S_y|} \in [0,1]$$

Where S_x = items rated by user x

**Cosine Similarity**
$$\text{Sim}(x,y) = \frac{r_x^T r_y}{||r_x|| \cdot ||r_y||} \in [0,1]$$

(Treat blanks as 0)

**Normalized Cosine Similarity**
$$\text{Sim}(x,y) = \frac{(r_x - \bar{r}_x)^T(r_y - \bar{r}_y)}{||r_x - \bar{r}_x|| \cdot ||r_y - \bar{r}_y||} \in [-1,1]$$

Where $\bar{r}_x$ = average rating of user x

**Step 2: Predict Rating**

For user x's rating on item i:

- **Naive Average**: 
$$r_{xi} = \frac{1}{|N(x,i)|} \sum_{y \in N(x,i)} r_{yi}$$

- **Weighted Average**:
$$r_{xi} = \frac{\sum_{y \in N(x,i)} \text{Sim}(x,y) \cdot r_{yi}}{\sum_{y \in N(x,i)} \text{Sim}(x,y)}$$

Where N(x,i) = {y ∈ N_x | r_yi exists}

### B. Item-Item CF

Similar to user-user, but find similar items instead:

$$r_{xi} = \frac{\sum_{j \in N(i,x)} \text{Sim}(i,j) \cdot r_{xj}}{\sum_{j \in N(i,x)} \text{Sim}(i,j)}$$

Where N(i,x) = {j ∈ N_i | r_xj exists}

## 3. Latent Factor Model

### Concept
Assume the utility matrix has low-rank structure:
$$R \approx WH$$

Where:
- **W** ∈ ℝ^(m×k): User-factor matrix (m users, k latent factors)
- **H** ∈ ℝ^(k×n): Item-factor matrix (k latent factors, n items)
- **k**: Number of latent factors (usually much smaller than m and n)

### Prediction
$$R_{ij} = W_{i·} H_{·j} = \sum_{t=1}^{k} W_{it} H_{tj}$$

### Optimization Problem
$$\min_{W,H} F(W,H) := \frac{1}{2}\sum_{(i,j)\in\Omega}(R_{ij} - W_{i·}H_{·j})^2 + \frac{\lambda}{2}(||W||_F^2 + ||H||_F^2)$$

Where λ ≥ 0 is ridge regularization parameter

### Gradient Descent
$$W^{(k+1)} \leftarrow W^{(k)} - \alpha_k \nabla_W F(W^{(k)}, H^{(k)})$$
$$H^{(k+1)} \leftarrow H^{(k)} - \alpha_k \nabla_H F(W^{(k)}, H^{(k)})$$

### Gradients
$$[\nabla_W F]_{it} = -\sum_{j:(i,j)\in\Omega}(R_{ij} - W_{i·}H_{·j})H_{tj} + \lambda W_{it}$$

$$[\nabla_H F]_{tj} = -\sum_{i:(i,j)\in\Omega}(R_{ij} - W_{i·}H_{·j})W_{it} + \lambda H_{tj}$$

### Stochastic Gradient Descent
Evaluate gradients on single ratings instead of all, faster per step but needs more iterations

## 4. Matrix Completion

### Problem Definition
Given partially observed matrix M with known entries in Ω, recover the complete matrix:

$$\min_X \text{rank}(X) \quad \text{s.t.} \quad X_{ij} = M_{ij}, \forall(i,j) \in \Omega$$

**This is NP-hard!**

### Convex Relaxation using Nuclear Norm

Since rank minimization is **NP-hard** (rank is non-convex), we replace it with its best convex approximation — the nuclear norm:

$$\min_X \text{rank}(X) \quad \Rightarrow \quad \min_X \|X\|_*$$
$$\text{s.t.} \quad X_{ij} = M_{ij} \qquad\qquad \text{s.t.} \quad X_{ij} = M_{ij}$$

**Why it works**: Minimizing $\|X\|_* = \sum_i \sigma_i$ acts as an L1 penalty on singular values, pushing many to zero and reducing rank — analogous to how L1 promotes sparsity in vectors.

**Theoretical backing**: Nuclear norm is the **convex envelope** of rank on $\{X \mid \sigma_1(X) \leq 1\}$ — the tightest possible convex approximation. Solved efficiently via SDP (Algorithm 1) or Proximal Gradient (Algorithm 2).

## 5. Rank & Nuclear Norm

### Rank Definition
For X ∈ ℝ^(m×n), rank(X) is:
- Dimension of column space
- Dimension of row space  
- Smallest k such that X = WH (W ∈ ℝ^(m×k), H ∈ ℝ^(k×n))
- Number of non-zero singular values

### Singular Value Decomposition (SVD)
$$X = U\Sigma V^T$$

Where:
- **U** ∈ ℝ^(m×m): Orthogonal, columns are left singular vectors
- **Σ** ∈ ℝ^(m×n): Diagonal, entries σ_i are singular values (σ_1 ≥ σ_2 ≥ ... ≥ σ_k ≥ 0)
- **V** ∈ ℝ^(n×n): Orthogonal, columns are right singular vectors

Compact form: $X = \sum_{i=1}^{k} \sigma_i u_i v_i^T$

### Nuclear Norm Definition
$$||X||_* = \sum_{i=1}^{\min(m,n)} \sigma_i(X)$$

(Sum of all singular values)

### Properties
- **Convex** (is a norm)
- **Non-convex counterpart**: rank(X) is non-convex
- **Convex envelope**: Nuclear norm is the convex envelope of rank on {X | σ_1(X) ≤ 1}

### Convex Relaxation
Replace rank minimization with nuclear norm minimization:

$$\min_X ||X||_* \quad \text{s.t.} \quad X_{ij} = M_{ij}, \forall(i,j) \in \Omega$$

## 6. Recovery Guarantee (Informal)

**Theorem**: Under certain assumptions, if:
- True matrix rank = k
- Observed entries uniformly at random
- Number of observations ≥ C·n·k·log²(n)

Then nuclear norm minimization recovers the true matrix with high probability.

## 7. Semidefinite Program (SDP)

### Standard SDP Form
$$\min_{X \in S^n} \langle C, X \rangle$$
$$\text{s.t.} \quad \langle A_i, X \rangle = b_i, \quad i \in [m]$$
$$X \succeq 0$$

Where:
- **C, A_i** ∈ S^n (symmetric matrices)
- **b** ∈ ℝ^m
- **X ≻ 0** means X is positive semidefinite

### SDP Dual
$$\max_{y \in \mathbb{R}^m} \langle b, y \rangle$$
$$\text{s.t.} \quad \sum_{i=1}^m y_i A_i \preceq C$$

### Weak Duality
For any primal feasible X and dual feasible y:
$$\langle C, X \rangle \geq \langle b, y \rangle$$

### Strong Duality
If both primal and dual are strictly feasible, optimal values match.

## 8. ALGORITHM 1: SDP-based Matrix Completion

### Nuclear Norm as SDP
$$||X||_* = \min_{W_1, W_2} \frac{1}{2}(\text{Tr}(W_1) + \text{Tr}(W_2))$$
$$\text{s.t.} \quad \begin{bmatrix} W_1 & X \\ X^T & W_2 \end{bmatrix} \preceq 0$$

### Matrix Completion SDP
$$\min_{W_1, W_2, X} \frac{1}{2}(\text{Tr}(W_1) + \text{Tr}(W_2))$$
$$\text{s.t.} \quad X_{ij} = M_{ij}, \quad (i,j) \in \Omega$$
$$\begin{bmatrix} W_1 & X \\ X^T & W_2 \end{bmatrix} \preceq 0$$

Can be solved by SDP solvers (SDPT3, Mosek, CVX)

## 9. ALGORITHM 2: PROXIMAL GRADIENT (PG)

### Penalized Problem
$$\min_X ||X||_* + \frac{1}{2\mu}\sum_{(i,j)\in\Omega}(X_{ij} - M_{ij})^2$$

Can rewrite as:
$$\min_X \underbrace{\frac{1}{2}\sum_{(i,j)\in\Omega}(X_{ij} - M_{ij})^2}_{f(X) \text{ smooth}} + \underbrace{\mu||X||_*}_{g(X) \text{ nonsmooth}}$$

### Gradient of f(X)
$$[\nabla f(X)]_{ij} = \begin{cases} X_{ij} - M_{ij} & \text{if } (i,j) \in \Omega \\ 0 & \text{otherwise} \end{cases}$$

### Proximal Mapping of Nuclear Norm
$$P_{\mu||\cdot||_*}(Y) = \arg\min_X \left\{\mu||X||_* + \frac{1}{2}||X-Y||_F^2\right\}$$

Computed via soft-thresholding of singular values:

1. Compute SVD: $Y = U\text{Diag}(\sigma)V^T$
2. Soft-threshold: $\gamma_i = \max(\sigma_i - \mu, 0)$
3. Result: $P_{\mu||\cdot||_*}(Y) = U\text{Diag}(\gamma)V^T$

### PG Algorithm
```
Input: M, μ, step size α
Initialize: X ← 0
For t = 1, 2, ...:
  G ← ∇f(X)
  X ← P_{μ||·||_*}(X - α·G)
Output: X
```

**Note**: Each iteration requires SVD, which is expensive for large matrices.

## 10. Netflix Prize model

### Dataset
- **Training**: 100,480,507 ratings from 480,189 users on 17,770 movies
- **Testing**: 2,817,131 ratings (hidden)

### Evaluation Metric
$$\text{RMSE} = \sqrt{\sum_{(i,j) \in \text{Test}} (R_{ij} - R_{ij}^*)^2}$$

### Baselines
- **Trivial** (average): RMSE = 1.0540
- **Netflix Cinematch**: RMSE = 0.9514 (10% improvement)
- **Grand Prize**: RMSE ≤ 0.8563 (10% improvement over Cinematch)

## 11. KEY TAKEAWAYS

| Method | Pros | Cons |
|--------|------|------|
| User-User CF | Simple, interpretable | Doesn't scale well, new user problem |
| Item-Item CF | Works well in practice | Requires similar items to exist |
| Latent Factor | Scalable, handles cold start better | Needs optimization, non-convex |
| Matrix Completion (NucNorm) | Theoretical guarantees, convex | Computationally expensive (SDP/PG) |

## 12. QUICK REFERENCE: FORMULAS FOR CALCULATIONS

**Jaccard**: $\frac{\|A \cap B\|}{\|A \cup B\|}$

**Cosine**: $\frac{a \cdot b}{\|a\| \|b\|}$

**Normalized Cosine**: $\frac{(a-\bar{a}) \cdot (b-\bar{b})}{\|a-\bar{a}\| \|b-\bar{b}\|}$

**Weighted Average**: $\frac{\sum w_i x_i}{\sum w_i}$

**SVD Reconstruction**: $X = U\Sigma V^T = \sum_i \sigma_i u_i v_i^T$

**Nuclear Norm**: $\|X\|_* = \sum_i \sigma_i(X)$

**Soft Threshold**: $\max(\sigma - \mu, 0)$